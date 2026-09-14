"""Tests for the gradio comparison app (skipped when gradio is missing).

``test_compare_stitch_deadlock_regression`` runs the actual button handler with
fake engines under a timeout, so the non-reentrant-lock deadlock in
``get_engine()`` cannot come back unnoticed. The ``@pytest.mark.e2e`` test runs
the real handler (checkpoint + GPU required).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("gradio")

from PIL import Image  # noqa: E402

import app  # noqa: E402


def _run_with_timeout(fn, timeout: float):
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"handler did not finish within {timeout}s (deadlock?)")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _fake_match(count: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], [32, 24], (count, 2)).astype(np.float32)
    return {
        "keypoints0": points,
        "keypoints1": points + 1.0,
        "descriptors0": np.zeros((count, 128), np.float32),
        "descriptors1": np.zeros((count, 128), np.float32),
    }


def _fake_warps():
    warped0 = np.zeros((24, 32, 3), np.uint8)
    warped0[:, :] = (30, 60, 200)
    warped1 = np.zeros((24, 32, 3), np.uint8)
    warped1[:, :] = (200, 60, 30)
    mask = np.full((24, 32), 255, np.uint8)
    return (warped0, warped1), (mask, mask)


class _FakeEngine:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def match(self, left, right, *, max_side=None):
        return _fake_match()

    def stitch_with_match(self, left, right, match, *, max_out_height=2400):
        warps, masks = _fake_warps()
        return {
            "method": "unistitch",
            "flag_check": True,
            "warped_bgr": warps,
            "masks": masks,
            "matches": len(match["keypoints0"]),
            "ssim": 0.9,
            "psnr": 30.0,
            "overlap_fraction": 1.0,
            "network_seconds": 0.1,
        }


def _fake_classical_pair(left, right, match, **kwargs):
    warps, masks = _fake_warps()
    return {
        "method": "classical",
        "flag_check": True,
        "warped_bgr": warps,
        "masks": masks,
        "inliers": np.ones(len(match["keypoints0"]), bool),
        "homography": np.eye(3),
        "canvas_size": (32, 24),
        "ssim": 0.91,
        "psnr": 31.0,
        "overlap_fraction": 1.0,
        "align_seconds": 0.01,
        "warp_seconds": 0.02,
    }


def test_output_components_serve_png():
    assert app.classical_image.format == "png"
    assert app.unistitch_image.format == "png"


def test_api_info_generation():
    assert app.demo.get_api_info()


def test_sample_images_exist():
    for path in (app.SAMPLE_LEFT, app.SAMPLE_RIGHT):
        assert path.is_file(), f"missing sample: {path}"
        with Image.open(path) as image:
            assert image.width > 100 and image.height > 100


def test_load_sample_returns_bundled_paths():
    left, right = app.load_sample()
    assert left == str(app.SAMPLE_LEFT)
    assert right == str(app.SAMPLE_RIGHT)


def test_describe_processing_shows_processing_size():
    left = Image.new("RGB", (720, 540))
    right = Image.new("RGB", (562, 540))
    text = app.describe_processing(left, right, 512)
    assert "720×540" in text and "562×540" in text
    assert "512×384" in text
    assert "Out-of-Memory" not in text


def test_describe_processing_warns_for_large_max_side():
    left = Image.new("RGB", (720, 540))
    right = Image.new("RGB", (562, 540))
    text = app.describe_processing(left, right, app.MAX_MAX_SIDE)
    assert "Out-of-Memory" in text


def test_describe_processing_without_inputs():
    assert "入力してください" in app.describe_processing(None, None, app.DEFAULT_MAX_SIDE)


def test_mark_stale_clears_results():
    values = app.mark_stale()
    assert values[0] is None and values[2] is None and values[4] is None
    assert values[5] == {}
    assert "再合成" in values[-1]


def test_select_diagnostic():
    diags = {"採用領域": np.zeros((4, 4, 3), np.uint8)}
    assert app.select_diagnostic("採用領域", diags) is diags["採用領域"]
    assert app.select_diagnostic("対応点", diags) is diags["採用領域"]  # fallback
    assert app.select_diagnostic("採用領域", {}) is None


def test_compare_stitch_deadlock_regression(monkeypatch):
    """The button handler must initialise the engine without a nested lock."""
    monkeypatch.setattr(app, "UniStitcher", _FakeEngine)
    monkeypatch.setattr(app, "_ENGINE", None)
    monkeypatch.setattr(app, "warp_classical_pair", _fake_classical_pair)

    outputs = _run_with_timeout(
        lambda: app.compare_stitch(Image.new("RGB", (64, 48)), Image.new("RGB", (64, 48)), 512, "採用領域"),
        timeout=20,
    )
    classical_image, classical_status, unistitch_image, unistitch_status, diag_image, diags, shared = outputs
    assert classical_image.size == (32, 24)
    assert unistitch_image.size == (32, 24)
    assert "古典手法" in classical_status and "インライア" in classical_status
    assert "UniStitch" in unistitch_status
    assert set(app.DIAG_VIEWS) <= set(diags)
    assert diag_image is diags["採用領域"]
    assert "共通マッチング" in shared


@pytest.mark.e2e
def test_compare_stitch_handler_runs_end_to_end():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available")
    from pair_inference import resolve_checkpoint

    try:
        resolve_checkpoint(None)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    left = Image.open(app.SAMPLE_LEFT).convert("RGB")
    right = Image.open(app.SAMPLE_RIGHT).convert("RGB")
    outputs = _run_with_timeout(
        lambda: app.compare_stitch(left, right, 512, "採用領域"),
        timeout=600,
    )
    classical_image, classical_status, unistitch_image, unistitch_status, diag_image, diags, shared = outputs
    assert classical_image is not None, classical_status
    assert unistitch_image is not None, unistitch_status
    assert "mSSIM" in classical_status and "mSSIM" in unistitch_status
    assert diag_image is not None
    assert "共通マッチング" in shared
