"""Tests for the classical baseline and the shared seam helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "Codes", ROOT / "keypoint_tool"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("onnxruntime")

from classical_stitch import estimate_homography, stitch_classical  # noqa: E402
from stitch_common import resize_pair, seam_compose  # noqa: E402


def test_resize_pair_keeps_common_scale():
    image0 = np.zeros((540, 720, 3), np.uint8)
    image1 = np.zeros((270, 180, 3), np.uint8)
    resized0, resized1, scale = resize_pair(image0, image1, 360)
    assert scale == pytest.approx(0.5)
    assert resized0.shape[:2] == (270, 360)
    assert resized1.shape[:2] == (135, 90)


def test_seam_compose_uses_one_source_per_pixel():
    height, width = 80, 160
    image0 = np.zeros((height, width, 3), np.uint8)
    image0[:, :] = (30, 60, 200)
    image1 = np.zeros((height, width, 3), np.uint8)
    image1[:, :] = (200, 60, 30)
    mask0 = np.full((height, width), 255, np.uint8)
    mask1 = np.full((height, width), 255, np.uint8)

    composite, selection0, selection1 = seam_compose(image0, image1, mask0, mask1)

    from_source0 = (composite == image0).all(axis=2)
    from_source1 = (composite == image1).all(axis=2)
    assert (from_source0 | from_source1).all(), "composite mixed pixels"
    assert not (from_source0 & from_source1).any(), "composite averaged pixels"
    assert ((selection0 > 0) ^ (selection1 > 0)).all(), "selections are not complementary"
    assert ((selection0 > 0) | (selection1 > 0)).all()


def test_estimate_homography_recovers_translation():
    rng = np.random.default_rng(0)
    points1 = rng.uniform([0, 0], [640, 480], (200, 2))
    homography = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -12.0], [2e-5, 1e-5, 1.0]])
    points0 = cv2.perspectiveTransform(points1[None].astype(np.float64), homography)[0]
    points0 += rng.normal(0, 0.5, points0.shape)

    estimated, inliers = estimate_homography(points0.astype(np.float32), points1.astype(np.float32))
    assert inliers.mean() > 0.9
    projected = cv2.perspectiveTransform(points1[None].astype(np.float64), estimated)[0]
    error = np.linalg.norm(projected - points0, axis=1)
    assert np.median(error) < 1.0


@pytest.mark.e2e
def test_classical_stitch_sample_pair():
    from onnx_models import RaCoAlikedLightGlue

    left = cv2.cvtColor(cv2.imread(str(ROOT / "samples/left.png")), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(str(ROOT / "samples/right.png")), cv2.COLOR_BGR2RGB)
    left, right, _ = resize_pair(left, right, 768)

    matcher = RaCoAlikedLightGlue(num_threads=8, providers=["CPUExecutionProvider"])
    match = matcher.match_pair(left, right)
    assert len(match["keypoints0"]) > 100

    result = stitch_classical(left, right, match)
    assert result["flag_check"]
    assert result["inliers"].sum() > 50
    composite = result["composite_bgr"]
    assert composite.shape[0] >= left.shape[0] * 0.8
    assert composite.std() > 10.0

    selection0, selection1 = result["selection"]
    warped0, warped1 = result["warped_bgr"]
    chosen0 = selection0 > 0
    chosen1 = selection1 > 0
    assert np.array_equal(composite[chosen0], warped0[chosen0])
    assert np.array_equal(composite[chosen1], warped1[chosen1])

    masks0, masks1 = result["masks"]
    union = (masks0 > 0) | (masks1 > 0)
    valid = chosen0 | chosen1
    assert np.array_equal(valid, union)
