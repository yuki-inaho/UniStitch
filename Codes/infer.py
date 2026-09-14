#!/usr/bin/env python3
"""Hydra-managed UniStitch inference / qualitative evaluation.

Runs a trained checkpoint over one or more dataset folders (``input1/``,
``input2/`` and ``<keypoint>_lmdb``, see ``Codes/dataset.py``), writes the
fused stitched images and reports overlap SSIM/PSNR.

Examples::

    # quick look at 10 UDIS-D testing pairs using the released checkpoint
    pixi run infer checkpoint=model_homo_stage2/epoch_best_model.pth limit=10

    # evaluate the ALIKED fine-tuned checkpoint
    pixi run infer data.keypoint=aliked data.descriptor_pad_dim=256 \
        checkpoint=outputs/.../checkpoints/best.pth limit=100
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

CODES_DIR = Path(__file__).resolve().parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))


def mask_ssim(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    _, ssim = structural_similarity(image1 * mask, image2 * mask, data_range=255, channel_axis=2, full=True)
    return float(np.sum(ssim * mask) / (np.sum(mask) + 1e-6))


def mask_psnr(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    image1 = image1 * mask / 255.0
    image2 = image2 * mask / 255.0
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / mask.sum())
    return float(20 * np.log10(1.0 / rmse))


def load_checkpoint(net: Any, path: str, torch_module: Any) -> None:
    from checkpoint_utils import load_model_state

    checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_model_state(net, state)
    print(f"checkpoint: {path}")


def run_dataset(
    net: Any,
    dataset: Any,
    output_dir: Path,
    *,
    limit: int | None,
    max_out_height: int,
    device: Any,
    torch_module: Any,
) -> dict[str, float]:
    from network import build_output_model

    ssim_list: list[float] = []
    psnr_list: list[float] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    torch = torch_module
    net.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            if limit is not None and index >= limit:
                break
            batch_value = dataset[index]
            inputs = [value.float().unsqueeze(0).to(device) for value in batch_value[:6]]
            batch_out, flag_check = build_output_model(net, *inputs, max_out_height=max_out_height)
            if not flag_check:
                print(f"[{output_dir.name}] index {index}: warp too large, skipped")
                continue
            output_tps_ref = batch_out["output_tps_ref"]
            output_tps_tgt = batch_out["output_tps_tgt"]
            output_ref = (output_tps_ref[0, 0:3].detach().cpu().numpy() * 127.5).transpose(1, 2, 0)
            output_tgt = (output_tps_tgt[0, 0:3].detach().cpu().numpy() * 127.5).transpose(1, 2, 0)
            overlap_mask = (output_tps_ref[0, 3:6] * output_tps_tgt[0, 3:6]).detach().cpu().numpy().transpose(1, 2, 0)
            ssim = mask_ssim(output_ref, output_tgt, overlap_mask)
            psnr = mask_psnr(output_ref, output_tgt, overlap_mask)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            fused = output_ref * (output_ref / (output_ref + output_tgt + 1e-6)) + output_tgt * (
                output_tgt / (output_ref + output_tgt + 1e-6)
            )
            cv2.imwrite(str(output_dir / f"{index + 1:06d}.jpg"), np.clip(fused, 0, 255).astype(np.uint8))
            print(f"[{output_dir.name}] index {index}: ssim={ssim:.4f} psnr={psnr:.4f}")
    summary = {
        "count": float(len(ssim_list)),
        "ssim": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "psnr": float(np.mean(psnr_list)) if psnr_list else float("nan"),
    }
    print(f"[{output_dir.name}] mean ssim={summary['ssim']:.4f} psnr={summary['psnr']:.4f} (n={len(ssim_list)})")
    return summary


def run(cfg: DictConfig) -> None:
    os.environ.setdefault("CUDA_DEVICES_ORDER", "PCI_BUS_ID")
    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    import torch
    from dataset import TestDataset
    from network import Network

    run_dir = Path(cfg.output_dir) if cfg.output_dir else Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run directory: {run_dir}\ndevice: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    net = Network(descriptor_dim=cfg.model.descriptor_dim)
    if torch.cuda.is_available():
        net = net.cuda()
    load_checkpoint(net, cfg.checkpoint, torch)

    summaries: dict[str, dict[str, float]] = {}
    for name, path in (("udis", cfg.data.test_path), ("others", cfg.data.test_others_path)):
        if not path or not Path(path).is_dir():
            print(f"dataset '{name}' not found at {path}; skipping")
            continue
        dataset = TestDataset(
            data_path=path,
            max_points=cfg.data.max_points,
            keypoint=cfg.data.keypoint,
            descriptor_pad_dim=cfg.data.descriptor_pad_dim,
            resize_max_side=cfg.get("max_side", None),
        )
        summaries[name] = run_dataset(
            net,
            dataset,
            run_dir / name,
            limit=cfg.limit,
            max_out_height=cfg.max_out_height,
            device=device,
            torch_module=torch,
        )

    (run_dir / "metrics.json").write_text(json.dumps(summaries, indent=2))
    print(f"metrics written to {run_dir / 'metrics.json'}")


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
