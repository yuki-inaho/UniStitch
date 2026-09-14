"""Tests for the in-process pair inference pipeline (``Codes/pair_inference.py``).

The unit tests run anywhere; the end-to-end stitch test needs the fine-tuned
checkpoint (``pixi run download-models``) and a CUDA GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "Codes", ROOT / "keypoint_tool"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("onnxruntime")

from pair_inference import (  # noqa: E402
    DEFAULT_MAX_POINTS,
    DESCRIPTOR_DIM,
    UniStitcher,
    prepare_batch,
    resolve_checkpoint,
)


def _require_checkpoint() -> Path:
    try:
        return resolve_checkpoint(None)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available")


def _fake_match(num_points: int = 120, width: int = 320, height: int = 240) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    points0 = rng.uniform([0, 0], [width - 1, height - 1], (num_points, 2)).astype(np.float32)
    points1 = points0 + rng.normal(0, 2, (num_points, 2)).astype(np.float32)
    descriptors0 = rng.normal(0, 1, (num_points, 128)).astype(np.float32)
    descriptors1 = rng.normal(0, 1, (num_points, 128)).astype(np.float32)
    return {
        "keypoints0": points0,
        "keypoints1": points1,
        "descriptors0": descriptors0,
        "descriptors1": descriptors1,
    }


@pytest.fixture(scope="module")
def sample_pair() -> tuple[np.ndarray, np.ndarray]:
    import cv2

    left = cv2.cvtColor(cv2.imread(str(ROOT / "samples/left.png")), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(str(ROOT / "samples/right.png")), cv2.COLOR_BGR2RGB)
    assert left is not None and right is not None
    return left, right


def test_prepare_batch_shapes_and_normalisation():
    image0 = np.zeros((240, 320, 3), np.uint8)
    image1 = np.zeros((240, 320, 3), np.uint8)
    batch = prepare_batch(image0, image1, _fake_match())

    assert len(batch) == 6
    assert tuple(batch[0].shape) == (1, 3, 240, 320)
    assert tuple(batch[2].shape) == (1, DEFAULT_MAX_POINTS, 2)
    assert tuple(batch[4].shape) == (1, DEFAULT_MAX_POINTS, DESCRIPTOR_DIM)
    for values in batch[2][0]:
        assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0
    # descriptors: first 128 columns kept, the rest zero-padded
    assert torch.count_nonzero(batch[4][0, :, 128:]) == 0


def test_prepare_batch_is_deterministic():
    image0 = np.zeros((240, 320, 3), np.uint8)
    image1 = np.zeros((240, 320, 3), np.uint8)
    first = prepare_batch(image0, image1, _fake_match())
    second = prepare_batch(image0, image1, _fake_match())
    for lhs, rhs in zip(first, second, strict=True):
        assert torch.equal(lhs, rhs)


def test_prepare_batch_resizes_right_image_to_left_size():
    match = _fake_match(width=320, height=240)
    batch = prepare_batch(
        np.zeros((240, 320, 3), np.uint8),
        np.zeros((120, 160, 3), np.uint8),
        match,
    )
    assert tuple(batch[1].shape) == (1, 3, 240, 320)


@pytest.mark.e2e
def test_stitch_sample_pair(sample_pair):
    _require_checkpoint()
    _require_cuda()

    left, right = sample_pair
    stitcher = UniStitcher(num_threads=8, providers=["CPUExecutionProvider"])
    result = stitcher.stitch(left, right, max_side=768)

    assert result["flag_check"], "warp size check failed"
    assert result["matches"] > 100
    width, height = result["output_size"]
    assert width >= left.shape[1]
    assert height >= 0.8 * left.shape[0]
    fused = result["fused_rgb"].astype(np.float32)
    assert fused.mean() > 20.0, "stitched output is almost black"
    assert fused.std() > 10.0, "stitched output looks flat"
    assert result["ssim"] > 0.5
