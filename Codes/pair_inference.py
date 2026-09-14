"""In-process UniStitch inference for arbitrary RGB image pairs.

``Codes/infer.py`` evaluates dataset folders (``input1/`` + ``input2/`` plus an
LMDB keypoint database).  The gradio app and the tests need the same pipeline
for two images chosen at runtime, so this module wraps:

* the ONNX keypoint matcher (``keypoint_tool/onnx_models.py``),
* ``Codes/network.py`` (homography + TPS mesh + fusion masks), and
* ``Codes/checkpoint_utils.py`` (checkpoint loading, incl. the truncated stem).

Preprocessing follows ``Codes/dataset.py::TestDataset`` (BGR normalisation,
zero-padded descriptors, points normalised by ``size - 1``) and the fusion
follows ``Codes/infer.py`` (``* 127.5`` + overlap-weighted blend).

Example::

    from pair_inference import UniStitcher

    stitcher = UniStitcher()          # loads checkpoint + ONNX keypoints
    result = stitcher.stitch(left_rgb, right_rgb)
    Image.fromarray(result["fused_rgb"]).save("stitched.png")
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "Codes", REPO_ROOT / "keypoint_tool"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from checkpoint_utils import infer_descriptor_dim, load_model_state  # noqa: E402
from dataset import pad_descriptor_dim  # noqa: E402
from network import Network, build_output_model  # noqa: E402

MODEL_DIR = REPO_ROOT / "model_homo_stage2"
DEFAULT_CHECKPOINT = MODEL_DIR / "unistitch-aliked-zeropad-epoch2-ssim0.8649.pth"
FALLBACK_CHECKPOINTS = (
    MODEL_DIR / "unistitch-aliked-zeropad-epoch0.pth",
    MODEL_DIR / "epoch_best_model.pth",
)

DEFAULT_MAX_POINTS = 2000
DESCRIPTOR_DIM = 256
DEFAULT_MAX_SIDE = 1024
DEFAULT_MAX_OUT_HEIGHT = 2400


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Return an existing checkpoint path (explicit path first, then defaults)."""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint not found: {candidate}")
        return candidate
    for candidate in (DEFAULT_CHECKPOINT, *FALLBACK_CHECKPOINTS):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no UniStitch checkpoint found under model_homo_stage2/; run `pixi run download-models`")


def load_network(checkpoint_path: str | Path | None = None, device: str = "cuda") -> tuple[Any, int]:
    """Load a UniStitch checkpoint and return ``(net, descriptor_dim)``."""
    path = resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    descriptor_dim = infer_descriptor_dim(state)
    net = Network(descriptor_dim=descriptor_dim)
    load_model_state(net, state)
    net.eval()
    net.fuse()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniStitch warping but is not available")
    net = net.to(device)
    return net, descriptor_dim


def match_images(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    *,
    max_side: int | None = DEFAULT_MAX_SIDE,
    num_threads: int | None = None,
    providers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run the RaCo-ALIKED-LightGlue+ ONNX pipeline on an RGB pair."""
    from onnx_models import RaCoAlikedLightGlue

    matcher = RaCoAlikedLightGlue(num_threads=num_threads, providers=providers)
    return matcher.match_pair(image0_rgb, image1_rgb, max_side=max_side)


def _pad_or_truncate(
    points: np.ndarray,
    descriptors: np.ndarray,
    target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``TestDataset._pad_or_truncate_points`` with a deterministic seed."""
    if points.shape[0] == target:
        return points, descriptors
    if points.shape[0] == 0:
        raise RuntimeError("keypoint matching produced no matches")
    if points.shape[0] > target:
        return points[:target], descriptors[:target]
    repeat = np.random.default_rng(seed).integers(0, points.shape[0], target - points.shape[0])
    return (
        np.concatenate([points, points[repeat]], axis=0),
        np.concatenate([descriptors, descriptors[repeat]], axis=0),
    )


def prepare_batch(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    match: dict[str, np.ndarray],
    *,
    descriptor_dim: int = DESCRIPTOR_DIM,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Convert an RGB pair + matches into the 6-tensor network batch."""
    image0 = np.ascontiguousarray(image0_rgb)
    image1 = np.ascontiguousarray(image1_rgb)
    height0, width0 = image0.shape[:2]
    height1, width1 = image1.shape[:2]

    points0 = match["keypoints0"].astype(np.float32).copy()
    points1 = match["keypoints1"].astype(np.float32).copy()
    descriptors0 = match["descriptors0"].astype(np.float32)
    descriptors1 = match["descriptors1"].astype(np.float32)

    if (height1, width1) != (height0, width0):
        # TestDataset resizes input2 to input1's size; scale its keypoints too.
        image1 = cv2.resize(image1, (width0, height0), interpolation=cv2.INTER_AREA)
        points1[:, 0] *= width0 / width1
        points1[:, 1] *= height0 / height1
        height1, width1 = height0, width0

    points0, descriptors0 = _pad_or_truncate(points0, descriptors0, max_points, seed)
    points1, descriptors1 = _pad_or_truncate(points1, descriptors1, max_points, seed + 1)

    points0[:, 0] /= width0 - 1
    points0[:, 1] /= height0 - 1
    points1[:, 0] /= width1 - 1
    points1[:, 1] /= height1 - 1

    def chw(image_rgb: np.ndarray) -> torch.Tensor:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        chw_array = np.transpose((bgr / 127.5) - 1.0, [2, 0, 1])
        return torch.from_numpy(np.ascontiguousarray(chw_array))

    return [
        chw(image0).unsqueeze(0),
        chw(image1).unsqueeze(0),
        torch.from_numpy(points0).float().unsqueeze(0),
        torch.from_numpy(points1).float().unsqueeze(0),
        pad_descriptor_dim(torch.from_numpy(descriptors0).float(), descriptor_dim).unsqueeze(0),
        pad_descriptor_dim(torch.from_numpy(descriptors1).float(), descriptor_dim).unsqueeze(0),
    ]


def fuse_warps(output_tps_ref: torch.Tensor, output_tps_tgt: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Blend the two warped views (BGR order, same as ``Codes/infer.py``)."""
    ref = output_tps_ref[0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0) * 127.5
    tgt = output_tps_tgt[0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0) * 127.5
    fused = ref * (ref / (ref + tgt + 1e-6)) + tgt * (tgt / (ref + tgt + 1e-6))
    fused_bgr = np.clip(fused, 0, 255).astype(np.uint8)
    return fused_bgr, cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)


def overlap_metrics(output_tps_ref: torch.Tensor, output_tps_tgt: torch.Tensor) -> tuple[float, float]:
    """Masked SSIM/PSNR on the overlap, identical to ``Codes/infer.py``."""
    from skimage.metrics import structural_similarity

    ref = output_tps_ref[0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0) * 127.5
    tgt = output_tps_tgt[0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0) * 127.5
    mask = (output_tps_ref[0, 3:6] * output_tps_tgt[0, 3:6]).detach().float().cpu().numpy().transpose(1, 2, 0)
    _, ssim = structural_similarity(ref * mask, tgt * mask, data_range=255, channel_axis=2, full=True)
    ssim = float(np.sum(ssim * mask) / (np.sum(mask) + 1e-6))
    ref_n = ref * mask / 255.0
    tgt_n = tgt * mask / 255.0
    rmse = np.sqrt(np.sum((ref_n - tgt_n) ** 2) / (mask.sum() + 1e-6))
    psnr = float(20 * np.log10(1.0 / rmse)) if rmse > 0 else float("inf")
    return ssim, psnr


class UniStitcher:
    """Cached uniStitch network + ONNX matcher for repeated stitches."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cuda",
        num_threads: int | None = None,
        providers: list[str] | None = None,
    ) -> None:
        from onnx_models import RaCoAlikedLightGlue

        self.net, self.descriptor_dim = load_network(checkpoint_path, device=device)
        self.matcher = RaCoAlikedLightGlue(num_threads=num_threads, providers=providers)

    def stitch(
        self,
        image0_rgb: np.ndarray,
        image1_rgb: np.ndarray,
        *,
        max_side: int | None = DEFAULT_MAX_SIDE,
        max_out_height: int = DEFAULT_MAX_OUT_HEIGHT,
    ) -> dict[str, Any]:
        """Stitch an RGB pair; returns fused images, metrics and timing."""
        started = time.time()
        match = self.matcher.match_pair(image0_rgb, image1_rgb, max_side=max_side)
        num_matches = int(len(match["keypoints0"]))
        if num_matches == 0:
            raise RuntimeError("keypoint matching found no matches between the two images")

        inputs = prepare_batch(image0_rgb, image1_rgb, match, descriptor_dim=self.descriptor_dim)
        inputs = [value.to(next(self.net.parameters()).device) for value in inputs]
        with torch.no_grad():
            batch_out, flag_check = build_output_model(self.net, *inputs, max_out_height=max_out_height)
        if not flag_check:
            return {
                "flag_check": False,
                "matches": num_matches,
                "elapsed": time.time() - started,
            }
        output_ref = batch_out["output_tps_ref"]
        output_tgt = batch_out["output_tps_tgt"]
        fused_bgr, fused_rgb = fuse_warps(output_ref, output_tgt)
        ssim, psnr = overlap_metrics(output_ref, output_tgt)
        return {
            "flag_check": True,
            "fused_bgr": fused_bgr,
            "fused_rgb": fused_rgb,
            "matches": num_matches,
            "ssim": ssim,
            "psnr": psnr,
            "output_size": (fused_rgb.shape[1], fused_rgb.shape[0]),
            "elapsed": time.time() - started,
        }
