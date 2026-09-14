#!/usr/bin/env python3
"""Hydra-managed UniStitch training / fine-tuning.

The released UniStitch checkpoint was trained with 256-d SuperPoint
descriptors. To fine-tune it on ALIKED features (128-d), the dataset zero-pads
descriptors to ``data.descriptor_pad_dim`` (256) so every pretrained weight
loads, and the point backbone learns to use the first 128 descriptor channels.

Examples::

    # stage-2 fine-tune from the released checkpoint, AMUSE optimizer, GPU 1
    pixi run train checkpoint.pretrained=model_homo_stage2/epoch_best_model.pth

    # classic AdamW baseline
    pixi run train optim=adamw optim.lr=1e-4

    # resume a run
    pixi run train checkpoint.resume=outputs/2026-09-14/17-30-00/checkpoints/last.pth

TensorBoard::

    pixi run tensorboard        # serves outputs/ (default port 6006)
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from beartype import beartype
from checkpoint_utils import load_model_state
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

CODES_DIR = Path(__file__).resolve().parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))


def setup_seed(seed: int) -> None:
    import torch
    import torch.backends.cudnn as cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def mask_ssim(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    _, ssim = structural_similarity(image1 * mask, image2 * mask, data_range=255, channel_axis=2, full=True)
    return float(np.sum(ssim * mask) / (np.sum(mask) + 1e-6))


@beartype
def build_param_groups(net: Any, optim_cfg: DictConfig) -> list[dict[str, Any]]:
    """Split parameters into Muon (hidden matrices) and AdamW-style groups."""
    head_patterns = tuple(optim_cfg.get("head_param_patterns", []))
    muon_params: list[Any] = []
    other_params: list[Any] = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if any(pattern in name for pattern in head_patterns):
            other_params.append(param)
        elif param.ndim >= 2:
            muon_params.append(param)
        else:
            other_params.append(param)
    print(f"optimizer groups: muon={len(muon_params)} matrices, aux={len(other_params)} tensors")
    return [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": optim_cfg.lr,
            "momentum": optim_cfg.momentum,
            "weight_decay": optim_cfg.weight_decay,
            "aux_update_type": optim_cfg.aux_update_type,
        },
        {
            "params": other_params,
            "use_muon": False,
            "update_type": "adamw",
            "lr": optim_cfg.aux_lr,
            "beta2": optim_cfg.beta2,
            "eps": optim_cfg.eps,
            "weight_decay": optim_cfg.weight_decay,
        },
    ]


def build_optimizer(net: Any, optim_cfg: DictConfig, total_steps: int) -> tuple[Any, str, Any]:
    """Create the configured optimizer (AMUSE by default) and optional scheduler."""
    if optim_cfg.type == "amuse":
        from optim.amuse import AMUSE

        warmup_steps = max(1, int(round(total_steps * float(optim_cfg.warmup_ratio))))
        optimizer = AMUSE(
            build_param_groups(net, optim_cfg),
            weight_decay_at_y=optim_cfg.weight_decay_at_y,
            beta1=optim_cfg.beta1,
            rho=optim_cfg.rho,
            r=optim_cfg.r,
            weight_lr_power=optim_cfg.weight_lr_power,
            warmup_steps=warmup_steps,
        )
        print(f"AMUSE: warmup_steps={warmup_steps} (total steps {total_steps})")
        return optimizer, "amuse", None

    import torch

    parameters = [param for param in net.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optim_cfg.lr,
        betas=tuple(optim_cfg.betas),
        eps=optim_cfg.eps,
        weight_decay=optim_cfg.weight_decay,
    )
    scheduler = None
    if optim_cfg.get("scheduler") is not None and optim_cfg.scheduler.get("type") == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=optim_cfg.scheduler.gamma)
    return optimizer, "adamw", scheduler


def freeze_modules(net: Any, patterns: list[str]) -> None:
    for pattern in patterns:
        module = net
        for part in pattern.split("."):
            module = getattr(module, part)
        module.requires_grad_(False)
    trainable = sum(param.numel() for param in net.parameters() if param.requires_grad)
    total = sum(param.numel() for param in net.parameters())
    print(f"trainable parameters: {trainable:,}/{total:,}")


class BestCheckpoints:
    """Keep only the K best checkpoints by validation SSIM."""

    @beartype
    def __init__(self, directory: Path, keep: int) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, int(keep))
        self.entries: list[tuple[float, int, Path]] = []

    @beartype
    def offer(self, score: float, epoch: int, state: dict[str, Any], torch_module: Any) -> Path:
        path = self.directory / f"best_epoch{epoch:04d}_ssim{score:.4f}.pth"
        torch_module.save(state, path)
        self.entries.append((score, epoch, path))
        self.entries.sort(key=lambda entry: (-entry[0], entry[1]))
        for _, _, stale in self.entries[self.keep :]:
            stale.unlink(missing_ok=True)
        self.entries = self.entries[: self.keep]
        best = self.entries[0][2]
        link = self.directory / "best.pth"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(best.name)
        return path


def batch_to_device(batch: list[Any], device: Any) -> list[Any]:
    return [value.float().to(device) if hasattr(value, "to") else value for value in batch]


@beartype
def compute_loss(batch_out: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss

    overlap_loss = cal_lp_loss(
        batch_out["output_H_ref"],
        batch_out["output_H_tgt"],
        batch_out["output_tps_ref"],
        batch_out["output_tps_tgt"],
    )
    nonoverlap_loss = 10.0 * inter_grid_loss(batch_out["mesh_ref"]) + 10.0 * intra_grid_loss(batch_out["mesh_ref"])
    nonoverlap_loss = (
        nonoverlap_loss + 10.0 * inter_grid_loss(batch_out["mesh_tgt"]) + 10.0 * intra_grid_loss(batch_out["mesh_tgt"])
    )
    total_loss = overlap_loss + nonoverlap_loss + batch_out["df_loss"]
    return total_loss, {
        "overlap": float(overlap_loss.detach()),
        "nonoverlap": float(nonoverlap_loss.detach()),
        "df": float(batch_out["df_loss"].detach()),
    }


def validate(
    net: Any,
    loader: Any,
    *,
    max_batches: int,
    max_out_height: int,
    device: Any,
    writer: Any,
    global_step: int,
    tag: str,
    log_images: bool,
) -> float:
    import torch
    from network import build_output_model

    net.eval()
    ssim_list: list[float] = []
    with torch.no_grad():
        for index, batch_value in enumerate(loader):
            if index >= max_batches:
                break
            inputs = batch_to_device(list(batch_value[:6]), device)
            batch_out, flag_check = build_output_model(net, *inputs, max_out_height=max_out_height)
            if not flag_check:
                continue
            output_tps_ref = batch_out["output_tps_ref"]
            output_tps_tgt = batch_out["output_tps_tgt"]
            output_ref = (output_tps_ref[0, 0:3].detach().cpu().numpy() * 127.5).transpose(1, 2, 0)
            output_tgt = (output_tps_tgt[0, 0:3].detach().cpu().numpy() * 127.5).transpose(1, 2, 0)
            overlap_mask = (output_tps_ref[0, 3:6] * output_tps_tgt[0, 3:6]).detach().cpu().numpy().transpose(1, 2, 0)
            ssim_list.append(mask_ssim(output_ref, output_tgt, overlap_mask))
            if log_images and index == 0 and writer is not None:
                image = np.clip((output_ref + output_tgt) / 2.0, 0, 255).astype(np.uint8)
                writer.add_image(f"val/{tag}_overlap", image, global_step, dataformats="HWC")
    score = float(np.mean(ssim_list)) if ssim_list else float("nan")
    return score


def train(cfg: DictConfig) -> None:
    os.environ.setdefault("CUDA_DEVICES_ORDER", "PCI_BUS_ID")
    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    import torch
    from dataset import TestDataset, TrainDataset
    from network import Network, build_model
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    run_dir = Path(cfg.output_dir) if cfg.output_dir else Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"run directory: {run_dir}")

    setup_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # --- data -----------------------------------------------------------------
    train_data = TrainDataset(
        data_path=cfg.data.train_path,
        max_points=cfg.data.max_points,
        keypoint=cfg.data.keypoint,
        descriptor_pad_dim=cfg.data.descriptor_pad_dim,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        drop_last=True,
    )
    test_loaders = {}
    for name, path in (("udis", cfg.data.test_path), ("others", cfg.data.test_others_path)):
        if not path or not Path(path).is_dir():
            print(f"validation dataset '{name}' not found at {path}; skipping")
            continue
        test_loaders[name] = DataLoader(
            TestDataset(
                data_path=path,
                max_points=cfg.data.max_points,
                keypoint=cfg.data.keypoint,
                descriptor_pad_dim=cfg.data.descriptor_pad_dim,
                resize_max_side=cfg.val.get("max_side", None),
            ),
            batch_size=1,
            num_workers=2,
            shuffle=False,
            drop_last=False,
        )

    # --- model ----------------------------------------------------------------
    net = Network(descriptor_dim=cfg.model.descriptor_dim)
    if torch.cuda.is_available():
        net = net.cuda()
    freeze_modules(net, list(cfg.stage.get("freeze", [])))

    start_epoch = 0
    glob_iter = 0
    best_score = 0.0
    epochs_without_improvement = 0
    resume_checkpoint = None
    if cfg.checkpoint.resume:
        resume_checkpoint = torch.load(cfg.checkpoint.resume, map_location="cpu", weights_only=False)
        load_model_state(net, resume_checkpoint["model"], strict=False)
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        glob_iter = int(resume_checkpoint.get("glob_iter", 0))
        best_score = float(resume_checkpoint.get("best_ssim", 0.0))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        print(f"resumed from {cfg.checkpoint.resume} (epoch {start_epoch}, iter {glob_iter})")
    elif cfg.checkpoint.pretrained:
        checkpoint = torch.load(cfg.checkpoint.pretrained, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state(net, state, strict=bool(cfg.checkpoint.strict))
        print(f"warm start from {cfg.checkpoint.pretrained}")

    # --- optimizer ------------------------------------------------------------
    optimizer, optimizer_type, scheduler = build_optimizer(net, cfg.optim, len(train_loader) * int(cfg.epochs))
    is_amuse = optimizer_type == "amuse"
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.last_epoch = start_epoch
        print("resumed optimizer state")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if cfg.log.tensorboard else None
    if writer is not None:
        print(f"tensorboard logdir: {run_dir / 'tensorboard'}")

    keep_best = BestCheckpoints(run_dir / "checkpoints", cfg.save.keep_best_k)
    if cfg.save.keep_last:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # --- training loop ---------------------------------------------------------
    score_interval = max(1, int(cfg.log.every_n_steps))
    loss_sums = {"total": 0.0, "overlap": 0.0, "nonoverlap": 0.0, "df": 0.0}
    loss_count = 0
    start_time = time.time()

    if is_amuse:
        optimizer.train()
    net.train()

    try:
        for epoch in range(start_epoch, int(cfg.epochs)):
            print(f"start epoch {epoch}")
            if is_amuse:
                optimizer.train()
            net.train()
            lr_values = [group["lr"] for group in optimizer.param_groups]

            for batch_value in train_loader:
                inputs = batch_to_device(list(batch_value[:6]), device)
                optimizer.zero_grad(set_to_none=True)
                batch_out = build_model(net, *inputs, is_stage2=bool(cfg.stage.is_stage2))
                total_loss, parts = compute_loss(batch_out)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
                optimizer.step()

                loss_sums["total"] += float(total_loss.detach())
                loss_sums["overlap"] += parts["overlap"]
                loss_sums["nonoverlap"] += parts["nonoverlap"]
                loss_sums["df"] += parts["df"]
                loss_count += 1

                if glob_iter % score_interval == 0:
                    count = max(1, loss_count)
                    average = {key: value / count for key, value in loss_sums.items()}
                    print(
                        f"epoch {epoch} iter {glob_iter}: loss={average['total']:.4f} "
                        f"(overlap {average['overlap']:.4f}, nonoverlap {average['nonoverlap']:.4f}, "
                        f"df {average['df']:.4f}) grad_norm={float(grad_norm):.3f}"
                    )
                    if writer is not None:
                        for key, value in average.items():
                            writer.add_scalar(f"train/{key}", value, glob_iter)
                        for index, group in enumerate(optimizer.param_groups):
                            writer.add_scalar(f"train/lr_{index}", group["lr"], glob_iter)
                        if cfg.log.grad_norm:
                            writer.add_scalar("train/grad_norm", float(grad_norm), glob_iter)
                        if is_amuse:
                            writer.add_scalar(
                                "train/amuse_beta1", optimizer.param_groups[0].get("beta1", 0.0), glob_iter
                            )
                        if glob_iter % (score_interval * 10) == 0:
                            writer.add_image(
                                "train/output_H_ref",
                                (batch_out["output_H_ref"][0, 0:3].detach().clamp(-1, 1) + 1.0) / 2.0,
                                glob_iter,
                            )
                    loss_sums = {"total": 0.0, "overlap": 0.0, "nonoverlap": 0.0, "df": 0.0}
                    loss_count = 0
                glob_iter += 1

            print(f"epoch {epoch} done ({time.time() - start_time:.0f}s), lr={lr_values}")
            if scheduler is not None:
                scheduler.step()

            # --- validation + checkpointing (evaluate x weights for AMUSE) ------
            if is_amuse:
                optimizer.eval()
            net.eval()
            scores: dict[str, float] = {}
            if (epoch + 1) % int(cfg.val.every_n_epochs) == 0:
                for name, loader in test_loaders.items():
                    max_batches = int(cfg.val.udis_max_batches if name == "udis" else cfg.val.others_max_batches)
                    score = validate(
                        net,
                        loader,
                        max_batches=max_batches,
                        max_out_height=int(cfg.val.max_out_height),
                        device=device,
                        writer=writer,
                        global_step=epoch + 1,
                        tag=name,
                        log_images=bool(cfg.val.log_images),
                    )
                    scores[name] = score
                    print(f"epoch {epoch}: validation SSIM [{name}] = {score:.4f}")
                    if writer is not None:
                        writer.add_scalar(f"val/ssim_{name}", score, epoch + 1)

            selection = scores.get("udis", scores.get("others", float("nan")))
            early_stop_cfg = cfg.get("early_stop")
            min_delta = float(early_stop_cfg.get("min_delta", 0.0)) if early_stop_cfg else 0.0
            patience = int(early_stop_cfg.get("patience", 0)) if early_stop_cfg else 0
            state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "glob_iter": glob_iter,
                "best_ssim": max(best_score, selection if np.isfinite(selection) else 0.0),
                "epochs_without_improvement": epochs_without_improvement,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
            if np.isfinite(selection) and selection > best_score + min_delta:
                best_score = selection
                epochs_without_improvement = 0
                state["epochs_without_improvement"] = 0
                path = keep_best.offer(selection, epoch + 1, state, torch)
                print(f"saved best checkpoint: {path.name}")
            elif np.isfinite(selection):
                epochs_without_improvement += 1
                state["epochs_without_improvement"] = epochs_without_improvement
                if patience:
                    print(
                        f"epoch {epoch}: no improvement (+{min_delta}), patience {epochs_without_improvement}/{patience}"
                    )
            if writer is not None:
                writer.add_scalar("val/epochs_without_improvement", epochs_without_improvement, epoch + 1)
            if cfg.save.keep_last:
                torch.save(state, run_dir / "checkpoints" / "last.pth")

            if is_amuse:
                optimizer.train()

            if (
                early_stop_cfg is not None
                and early_stop_cfg.get("enabled", False)
                and patience
                and epochs_without_improvement >= patience
            ):
                print(
                    f"early stopping after epoch {epoch} ({epochs_without_improvement} validations without improvement)"
                )
                break
    except KeyboardInterrupt:
        print("interrupted; saving last.pth")
        if is_amuse:
            optimizer.eval()  # store the evaluation (averaged) iterate, as elsewhere
        if cfg.save.keep_last:
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": start_epoch,
                    "glob_iter": glob_iter,
                    "best_ssim": best_score,
                    "epochs_without_improvement": epochs_without_improvement,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                run_dir / "checkpoints" / "last.pth",
            )
    finally:
        if writer is not None:
            writer.close()
    print(f"training finished; best validation SSIM = {best_score:.4f}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
