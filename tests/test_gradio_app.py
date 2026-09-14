"""Tests for the gradio app layout/helpers (skipped when gradio is missing).

``test_stitch_images_handler_deadlock_regression`` runs the actual button
handler with a fake engine under a timeout, so the non-reentrant-lock deadlock
in ``get_engine()`` cannot come back unnoticed. The ``@pytest.mark.e2e`` test
additionally runs the real handler (checkpoint + GPU required).
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


def test_output_component_serves_png():
    assert app.output_image.format == "png"


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


def test_describe_processing_shows_sizes():
    left = Image.new("RGB", (720, 540))
    right = Image.new("RGB", (562, 540))
    text = app.describe_processing(left, right, app.DEFAULT_MAX_SIDE)
    assert "720×540" in text and "562×540" in text
    assert "1024" in text
    assert "Out-of-Memory" not in text


def test_describe_processing_warns_for_large_max_side():
    left = Image.new("RGB", (720, 540))
    right = Image.new("RGB", (562, 540))
    text = app.describe_processing(left, right, app.MAX_MAX_SIDE)
    assert "Out-of-Memory" in text


def test_describe_processing_without_inputs():
    assert "入力してください" in app.describe_processing(None, None, app.DEFAULT_MAX_SIDE)


def test_mark_stale_clears_result():
    image, message = app.mark_stale()
    assert image is None
    assert "再合成" in message


def test_stitch_images_handler_deadlock_regression(monkeypatch):
    """The button handler must initialise the engine without a nested lock."""

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stitch(self, left, right, *, max_side, **kwargs):
            return {
                "flag_check": True,
                "fused_rgb": np.zeros((24, 32, 3), np.uint8),
                "matches": 10,
                "ssim": 0.9,
                "psnr": 30.0,
                "output_size": (32, 24),
                "elapsed": 0.1,
            }

    monkeypatch.setattr(app, "UniStitcher", FakeEngine)
    monkeypatch.setattr(app, "_ENGINE", None)

    image, status = _run_with_timeout(
        lambda: app.stitch_images(Image.new("RGB", (64, 48)), Image.new("RGB", (64, 48)), 512),
        timeout=20,
    )
    assert isinstance(image, Image.Image)
    assert image.size == (32, 24)
    assert "合成結果" in status
    assert "対応点 10" in status


@pytest.mark.e2e
def test_stitch_images_handler_runs_end_to_end():
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
    image, status = _run_with_timeout(
        lambda: app.stitch_images(left, right, 768),
        timeout=300,
    )
    assert image.width > left.width * 0.5
    assert "合成結果" in status
    assert "対応点" in status
