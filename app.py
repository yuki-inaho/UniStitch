"""Gradio comparison demo: classical stitching vs UniStitch.

Both pipelines consume the *same* keypoint matching result (RaCo-ALIKED-
LightGlue+ ONNX) and the *same* finish (GraphCut seam cut, one source per
pixel), so the only difference is the alignment:

* classical: MAGSAC++ homography (OpenCV USAC_MAGSAC) + perspective warp
* UniStitch: homography + TPS mesh predicted by the fine-tuned network

Run with::

    pixi run download-models   # release checkpoints (once)
    pixi run app               # http://127.0.0.1:7860
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

from classical_stitch import warp_classical_pair  # noqa: E402
from pair_inference import UniStitcher  # noqa: E402
from stitch_common import (  # noqa: E402
    draw_matches,
    overlay_50,
    resize_pair,
    seam_compose,
    selection_overlay,
    side_by_side,
)

SAMPLE_LEFT = REPO_ROOT / "samples" / "left.png"
SAMPLE_RIGHT = REPO_ROOT / "samples" / "right.png"

DEFAULT_MAX_SIDE = 1024
MIN_MAX_SIDE = 384
MAX_MAX_SIDE = 1600
LARGE_MAX_SIDE_HINT = 1280

DIAG_VIEWS = ["採用領域", "変形A", "変形B", "50%重ね", "対応点"]
DIAG_DEFAULT = DIAG_VIEWS[0]

# Engine creation and inference use separate locks: get_engine() is called from
# inside compare_stitch(), so a single non-reentrant lock would deadlock.
_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_ENGINE: UniStitcher | None = None
_ENGINE_LOAD_SECONDS = 0.0


def get_engine() -> UniStitcher:
    global _ENGINE, _ENGINE_LOAD_SECONDS
    if _ENGINE is None:
        with _INIT_LOCK:
            if _ENGINE is None:
                started = time.time()
                _ENGINE = UniStitcher(num_threads=8)
                _ENGINE_LOAD_SECONDS = time.time() - started
    return _ENGINE


def processing_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_side) / float(max(width, height)))
    return max(16, round(width * scale)), max(16, round(height * scale))


def describe_processing(left, right, max_side):
    if left is None or right is None:
        return "**処理サイズ**: 左右の画像を入力してください。"
    left_size = processing_size(left.width, left.height, int(max_side))
    right_size = processing_size(right.width, right.height, int(max_side))
    note = ""
    if int(max_side) > LARGE_MAX_SIDE_HINT:
        note = "\n\n⚠️ 最大辺が大きいため、8 GiB GPU では Out-of-Memory になる可能性があります。"
    return (
        f"**処理サイズ**: 左 {left.width}×{left.height} → {left_size[0]}×{left_size[1]} / "
        f"右 {right.width}×{right.height} → {right_size[0]}×{right_size[1]}  |  "
        f"両方式に共通適用（長辺 ≤ {int(max_side)}px）{note}"
    )


def mark_stale():
    message = "入力または設定が変更されました。もう一度 **両方式で比較する** を押して再合成してください。"
    return None, "", None, "", None, {}, message


def _finish_with_seam(result: dict) -> dict:
    """Apply the shared GraphCut seam cut (no blending) and record its timing."""
    if not result.get("flag_check"):
        raise RuntimeError("合成領域が大きすぎて処理できません（最大辺を下げてください）")
    warped0, warped1 = result["warped_bgr"]
    mask0, mask1 = result["masks"]
    started = time.time()
    composite, selection0, selection1 = seam_compose(warped0, warped1, mask0, mask1)
    result["composite_bgr"] = composite
    result["composite_rgb"] = composite[:, :, ::-1].copy()
    result["selection"] = (selection0, selection1)
    result["seam_seconds"] = time.time() - started
    return result


def _classical_status(result: dict | None, error: str | None) -> str:
    if result is None:
        return f"**古典手法 + シーム**: 失敗 — {error}"
    height, width = result["composite_bgr"].shape[:2]
    inliers = int(result["inliers"].sum())
    total = int(len(result["inliers"]))
    return (
        f"**古典手法 + シーム**（LightGlue+ / MAGSAC++ / GraphCut）: {width}×{height}  |  "
        f"対応点 {total}（インライア {inliers}）  |  "
        f"mSSIM {result['ssim']:.4f} / mPSNR {result['psnr']:.2f}  |  "
        f"位置合わせ {result['align_seconds']:.2f}秒 + シーム {result['seam_seconds']:.1f}秒"
    )


def _unistitch_status(result: dict | None, error: str | None) -> str:
    if result is None:
        return f"**UniStitch + シーム**: 失敗 — {error}"
    height, width = result["composite_bgr"].shape[:2]
    return (
        f"**UniStitch + シーム**: {width}×{height}  |  対応点 {result['matches']}  |  "
        f"mSSIM {result['ssim']:.4f} / mPSNR {result['psnr']:.2f}  |  "
        f"ネットワーク {result['network_seconds']:.1f}秒 + シーム {result['seam_seconds']:.1f}秒"
    )


def _build_diagnostics(left_rgb, right_rgb, match, classical, unistitch) -> dict[str, np.ndarray]:
    diags: dict[str, np.ndarray] = {}
    methods = [("古典手法 + シーム", classical), ("UniStitch + シーム", unistitch)]

    selection_panels, selection_labels = [], []
    warped_a, warped_b, overlay_panels = [], [], []
    for label, result in methods:
        if result is None:
            continue
        selection_panels.append(selection_overlay(result["composite_bgr"], *result["selection"]))
        selection_labels.append(label)
        warped_a.append(result["warped_bgr"][0])
        warped_b.append(result["warped_bgr"][1])
        overlay_panels.append(overlay_50(result["warped_bgr"][0], result["warped_bgr"][1], *result["masks"]))

    if selection_panels:
        diags["採用領域"] = side_by_side(selection_panels, selection_labels)
        diags["変形A"] = side_by_side(warped_a, selection_labels)
        diags["変形B"] = side_by_side(warped_b, selection_labels)
        diags["50%重ね"] = side_by_side(overlay_panels, selection_labels)

    inliers = classical["inliers"] if classical is not None else None
    diags["対応点"] = draw_matches(left_rgb, right_rgb, match["keypoints0"], match["keypoints1"], inliers=inliers)
    return diags


def compare_stitch(left, right, max_side, view):
    """Run both pipelines on the same resized pair and shared matches."""
    if left is None or right is None:
        raise gr.Error("左画像・右画像の両方を入力してください。")

    left_rgb = np.asarray(left.convert("RGB"))
    right_rgb = np.asarray(right.convert("RGB"))
    left_rgb, right_rgb, _ = resize_pair(left_rgb, right_rgb, int(max_side))

    engine = get_engine()
    classical = None
    unistitch = None
    errors: dict[str, str] = {}
    try:
        with _INFER_LOCK:
            started = time.time()
            match = engine.match(left_rgb, right_rgb)
            match_seconds = time.time() - started
            if len(match["keypoints0"]) == 0:
                raise RuntimeError("対応点が見つかりませんでした。重なりの大きい画像を選んでください。")
            try:
                classical = _finish_with_seam(warp_classical_pair(left_rgb, right_rgb, match))
            except Exception as exc:  # noqa: BLE001 - reported per column
                errors["classical"] = str(exc)
            try:
                unistitch = _finish_with_seam(engine.stitch_with_match(left_rgb, right_rgb, match))
            except Exception as exc:  # noqa: BLE001 - reported per column
                errors["unistitch"] = str(exc)
    except FileNotFoundError as exc:
        raise gr.Error(
            f"チェックポイントが見つかりません ({exc})。`pixi run download-models` を実行してください。"
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise gr.Error(
                "CUDA out of memory: 処理解像度を下げるか、より大きな VRAM の GPU を使用してください。"
            ) from exc
        raise gr.Error(str(exc)) from exc

    classical_image = Image.fromarray(classical["composite_rgb"]) if classical is not None else None
    unistitch_image = Image.fromarray(unistitch["composite_rgb"]) if unistitch is not None else None
    diags = _build_diagnostics(left_rgb, right_rgb, match, classical, unistitch)
    diag_image = diags.get(view)
    if diag_image is None:
        diag_image = diags.get(DIAG_DEFAULT)
    shared = (
        f"共通マッチング {len(match['keypoints0'])}点 / {match_seconds:.1f}秒  |  "
        f"初回モデル読込 {_ENGINE_LOAD_SECONDS:.1f}秒  |  "
        f"処理解像度 {left_rgb.shape[1]}×{left_rgb.shape[0]} / {right_rgb.shape[1]}×{right_rgb.shape[0]}"
    )
    return (
        classical_image,
        _classical_status(classical, errors.get("classical")),
        unistitch_image,
        _unistitch_status(unistitch, errors.get("unistitch")),
        diag_image,
        diags,
        shared,
    )


def select_diagnostic(view, diags):
    if not diags:
        return None
    image = diags.get(view)
    if image is None:
        image = diags.get(DIAG_DEFAULT)
    return image


def load_sample():
    return str(SAMPLE_LEFT), str(SAMPLE_RIGHT)


with gr.Blocks(title="UniStitch: 古典手法との比較") as demo:
    gr.Markdown(
        "# UniStitch: Unified Image Stitching — 古典手法との比較\n"
        "同じ入力・同じ対応点・同じ継ぎ目処理（GraphCut シームカット）で、"
        "古典的な **MAGSAC++** と **UniStitch（ホモグラフィ+TPS）** を左右に並べて比較します。"
        "画素はブレンドせず、シームに従って片方から採用します。"
    )

    input_left = gr.Image(label="左画像 (A)", type="pil", height=240, render=False)
    input_right = gr.Image(label="右画像 (B)", type="pil", height=240, render=False)
    max_side = gr.Slider(
        MIN_MAX_SIDE,
        MAX_MAX_SIDE,
        value=DEFAULT_MAX_SIDE,
        step=64,
        label="処理解像度（長辺, px）",
        info="入力全体を縮小して両方式に同じ条件で適用します",
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
    compare_button = gr.Button("両方式で比較する", variant="primary")
    shared_info = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 古典手法 + シーム")
            classical_image = gr.Image(label="古典手法の合成結果 (PNG)", type="pil", format="png", height=420)
            classical_status = gr.Markdown()
        with gr.Column():
            gr.Markdown("### UniStitch + シーム")
            unistitch_image = gr.Image(label="UniStitch の合成結果 (PNG)", type="pil", format="png", height=420)
            unistitch_status = gr.Markdown()

    with gr.Accordion("位置合わせを確認", open=False):
        diag_view = gr.Radio(DIAG_VIEWS, value=DIAG_DEFAULT, label="表示")
        diag_image = gr.Image(label="診断表示", type="pil", height=440)

    with gr.Accordion("モデル・実行環境について", open=False):
        gr.Markdown(
            "- UniStitch: *Unifying Semantic and Geometric Features for Image Stitching* "
            "([paper](https://arxiv.org/abs/2603.10568))\n"
            "- キーポイント: RaCo-ALIKED-LightGlue+（vendored ONNX, CPU）を両方式で共有\n"
            "- 古典手法: OpenCV `USAC_MAGSAC`（ホモグラフィ）\n"
            "- 仕上げ: OpenCV `GraphCutSeamFinder`（共通）\n"
            "- 「50%重ね」は位置ずれ確認用の診断表示です（成果物はシーム合成）"
        )

    diag_state = gr.State({})

    sample_button.click(fn=load_sample, inputs=None, outputs=[input_left, input_right])
    compare_button.click(
        fn=compare_stitch,
        inputs=[input_left, input_right, max_side, diag_view],
        outputs=[
            classical_image,
            classical_status,
            unistitch_image,
            unistitch_status,
            diag_image,
            diag_state,
            shared_info,
        ],
    )
    diag_view.change(fn=select_diagnostic, inputs=[diag_view, diag_state], outputs=[diag_image])
    for component in (input_left, input_right, max_side):
        component.change(
            fn=describe_processing,
            inputs=[input_left, input_right, max_side],
            outputs=[size_info],
        )
        component.change(
            fn=mark_stale,
            inputs=None,
            outputs=[
                classical_image,
                classical_status,
                unistitch_image,
                unistitch_status,
                diag_image,
                diag_state,
                shared_info,
            ],
        )

if __name__ == "__main__":
    launch_kwargs = {"show_error": True}
    if os.environ.get("GRADIO_SHARE") == "1":
        launch_kwargs["share"] = True
    if os.environ.get("SPACE_ID"):  # Hugging Face Spaces
        launch_kwargs.update(server_name="0.0.0.0", server_port=7860)
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
