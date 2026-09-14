"""Gradio demo for UniStitch (unified semantic + geometric image stitching).

Run with::

    pixi run download-models   # fetch the release checkpoints (once)
    pixi run app               # http://127.0.0.1:7860

The app matches the two input images with the vendored ONNX keypoint pipeline
(RaCo-ALIKED-LightGlue+) and stitches them with the fine-tuned ALIKED
checkpoint (``model_homo_stage2/unistitch-aliked-zeropad-epoch2-ssim0.8649.pth``).
Layout follows the NIS gradio demo (samples on top, compact inputs, PNG output).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "Codes"))

from pair_inference import UniStitcher  # noqa: E402

SAMPLE_LEFT = REPO_ROOT / "samples" / "left.png"
SAMPLE_RIGHT = REPO_ROOT / "samples" / "right.png"

DEFAULT_MAX_SIDE = 1024
MIN_MAX_SIDE = 384
MAX_MAX_SIDE = 1600
LARGE_MAX_SIDE_HINT = 1280

# Engine creation and inference use separate locks: get_engine() is called from
# inside stitch_images(), so a single non-reentrant lock would deadlock.
_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_ENGINE: UniStitcher | None = None


def get_engine() -> UniStitcher:
    global _ENGINE
    if _ENGINE is None:
        with _INIT_LOCK:
            if _ENGINE is None:
                _ENGINE = UniStitcher(num_threads=8)
    return _ENGINE


def describe_processing(left, right, max_side):
    if left is None or right is None:
        return "**入力サイズ**: 左右の画像を入力してください。"
    note = ""
    if int(max_side) > LARGE_MAX_SIDE_HINT:
        note = "\n\n⚠️ 最大辺が大きいため、8 GiB GPU では Out-of-Memory になる可能性があります。"
    return (
        f"**入力サイズ**: 左 {left.width}×{left.height} / 右 {right.width}×{right.height}  |  "
        f"マッチングは長辺 ≤ {int(max_side)}px（合成は入力解像度）{note}"
    )


def mark_stale():
    return None, "入力または設定が変更されました。もう一度 **画像を合成する** を押して再合成してください。"


def stitch_images(left, right, max_side):
    """Stitch an RGB pair and return (panorama, status markdown)."""
    if left is None or right is None:
        raise gr.Error("左画像・右画像の両方を入力してください。")

    left_rgb = np.asarray(left.convert("RGB"))
    right_rgb = np.asarray(right.convert("RGB"))

    started = time.time()
    try:
        engine = get_engine()
        with _INFER_LOCK:
            result = engine.stitch(left_rgb, right_rgb, max_side=int(max_side))
    except FileNotFoundError as exc:
        raise gr.Error(
            f"チェックポイントが見つかりません ({exc})。`pixi run download-models` を実行してください。"
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise gr.Error("CUDA out of memory: 最大辺を下げるか、より大きな VRAM の GPU を使用してください。") from exc
        raise gr.Error(str(exc)) from exc

    if not result["flag_check"]:
        raise gr.Error("合成領域が大きすぎて処理できません。最大辺を下げるか、重なりの大きい画像を選んでください。")

    panorama = Image.fromarray(result["fused_rgb"])
    elapsed = time.time() - started
    status = (
        f"**合成結果**: {panorama.width}×{panorama.height}  |  対応点 {result['matches']}  |  "
        f"mSSIM {result['ssim']:.4f} / mPSNR {result['psnr']:.2f}  |  {elapsed:.1f}秒"
    )
    return panorama, status


def load_sample():
    return str(SAMPLE_LEFT), str(SAMPLE_RIGHT)


with gr.Blocks(title="UniStitch: Unified Image Stitching") as demo:
    gr.Markdown(
        "# UniStitch: Unified Image Stitching\n"
        "左右の画像を選び、**画像を合成する** を押すと合成します（結果は PNG で表示・保存できます）。"
    )

    input_left = gr.Image(label="左画像 (A)", type="pil", height=260, render=False)
    input_right = gr.Image(label="右画像 (B)", type="pil", height=260, render=False)
    max_side = gr.Slider(
        MIN_MAX_SIDE,
        MAX_MAX_SIDE,
        value=DEFAULT_MAX_SIDE,
        step=64,
        label="マッチングの最大辺 (px)",
        info="小さいほど省メモリ・高速（合成は入力解像度のまま）",
        render=False,
    )

    gr.Markdown("#### サンプルで試す")
    with gr.Row():
        gr.Image(
            value=str(SAMPLE_LEFT),
            label="サンプル左",
            interactive=False,
            height=110,
            show_download_button=False,
            show_fullscreen_button=False,
        )
        gr.Image(
            value=str(SAMPLE_RIGHT),
            label="サンプル右",
            interactive=False,
            height=110,
            show_download_button=False,
            show_fullscreen_button=False,
        )
    sample_button = gr.Button("サンプルを読み込む", size="sm")

    with gr.Row():
        input_left.render()
        input_right.render()

    size_info = gr.Markdown(describe_processing(None, None, DEFAULT_MAX_SIDE))

    max_side.render()
    stitch_button = gr.Button("画像を合成する", variant="primary")

    output_image = gr.Image(label="合成結果 (PNG)", type="pil", format="png", height=540)
    status_text = gr.Markdown()

    with gr.Accordion("モデル・実行環境について", open=False):
        gr.Markdown(
            "- UniStitch: *Unifying Semantic and Geometric Features for Image Stitching* "
            "([paper](https://arxiv.org/abs/2603.10568))\n"
            "- キーポイント: RaCo-ALIKED-LightGlue+ (vendored ONNX, CPU)\n"
            "- 合成: ALIKED 128-d zero-padded 256-d fine-tune "
            "(`unistitch-aliked-zeropad-epoch2-ssim0.8649.pth`)\n"
            "- 入力は重なりが必要です。大きくずれたペアや重なりのないペアは失敗します"
        )

    sample_button.click(fn=load_sample, inputs=None, outputs=[input_left, input_right])
    stitch_button.click(
        fn=stitch_images,
        inputs=[input_left, input_right, max_side],
        outputs=[output_image, status_text],
    )
    for component in (input_left, input_right, max_side):
        component.change(
            fn=describe_processing,
            inputs=[input_left, input_right, max_side],
            outputs=[size_info],
        )
        component.change(fn=mark_stale, inputs=None, outputs=[output_image, status_text])

if __name__ == "__main__":
    launch_kwargs = {"show_error": True}
    if os.environ.get("GRADIO_SHARE") == "1":
        launch_kwargs["share"] = True
    if os.environ.get("SPACE_ID"):  # Hugging Face Spaces
        launch_kwargs.update(server_name="0.0.0.0", server_port=7860)
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
