"""Shared helpers for the classical vs UniStitch stitching comparison.

Both pipelines in the comparison app share:

* one common input resize (the "処理解像度" slider bounds the whole pipeline),
* one shared keypoint matching result,
* the same finish: a GraphCut seam cut where every pixel is copied from exactly
  one of the two warped sources (no feathering / no alpha blending).

Everything in this module works on BGR ``uint8`` images (OpenCV convention).
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_pair(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    max_side: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resize both inputs by one common factor so the longest side fits ``max_side``."""
    height0, width0 = image0_rgb.shape[:2]
    height1, width1 = image1_rgb.shape[:2]
    longest = max(height0, width0, height1, width1)
    if max_side is None or longest <= max_side:
        return image0_rgb, image1_rgb, 1.0
    scale = float(max_side) / float(longest)

    def _resize(image: np.ndarray) -> np.ndarray:
        size = (max(16, round(image.shape[1] * scale)), max(16, round(image.shape[0] * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return _resize(image0_rgb), _resize(image1_rgb), scale


def warp_with_mask(
    image_bgr: np.ndarray,
    matrix: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image into ``canvas_size`` and return ``(image, mask)`` (mask 0/255)."""
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ones = np.full(image_bgr.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(
        ones,
        matrix,
        canvas_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, (mask > 0).astype(np.uint8) * 255


def overlap_mask(mask0: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    return (((mask0 > 0) & (mask1 > 0)).astype(np.uint8)) * 255


def seam_compose(
    warped0: np.ndarray,
    warped1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    corner0: tuple[int, int] = (0, 0),
    corner1: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GraphCut seam cut: each pixel comes from exactly one source.

    Returns ``(composite, selection0, selection1)`` where the selections are
    complementary 0/255 masks. If the seam finder cannot run (e.g. no overlap),
    the first image wins the overlapping region.
    """
    selection0 = (mask0 > 0).astype(np.uint8) * 255
    selection1 = (mask1 > 0).astype(np.uint8) * 255
    try:
        finder = cv2.detail.SeamFinder.createDefault(2)  # GC_COLOR_GRAD
        finder.find(
            [warped0, warped1],
            [tuple(int(v) for v in corner0), tuple(int(v) for v in corner1)],
            [selection0, selection1],
        )
    except cv2.error:
        pass
    # The seam finder may leave soft / overlapping boundaries: canonicalise to a
    # strict single-source selection (source 0 has priority in the overlap).
    chosen0 = selection0 > 0
    chosen1 = (selection1 > 0) & ~chosen0
    selection0 = chosen0.astype(np.uint8) * 255
    selection1 = chosen1.astype(np.uint8) * 255
    composite = compose_from_selection(warped0, warped1, selection0, selection1)
    return composite, selection0, selection1


def compose_from_selection(
    warped0: np.ndarray,
    warped1: np.ndarray,
    selection0: np.ndarray,
    selection1: np.ndarray,
) -> np.ndarray:
    """Copy each pixel from exactly one source (image1 wins only where selected)."""
    chosen1 = (selection1 > 0) & ~(selection0 > 0)
    composite = warped0.copy()
    composite[chosen1] = warped1[chosen1]
    return composite


def masked_ssim_psnr(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Masked SSIM/PSNR on the overlap, identical to ``Codes/infer.py``."""
    from skimage.metrics import structural_similarity

    if int((mask > 0).sum()) == 0:
        return float("nan"), float("nan")
    mask3 = np.broadcast_to((mask > 0)[..., None], image0_bgr.shape).astype(np.float32)
    _, ssim_map = structural_similarity(
        image0_bgr.astype(np.float32) * mask3,
        image1_bgr.astype(np.float32) * mask3,
        data_range=255,
        channel_axis=2,
        full=True,
    )
    ssim_value = float(np.sum(ssim_map * mask3) / (np.sum(mask3) + 1e-6))
    image0_n = image0_bgr.astype(np.float32) * mask3 / 255.0
    image1_n = image1_bgr.astype(np.float32) * mask3 / 255.0
    rmse = float(np.sqrt(np.sum((image0_n - image1_n) ** 2) / (mask3.sum() + 1e-6)))
    psnr_value = float(20 * np.log10(1.0 / rmse)) if rmse > 0 else float("inf")
    return ssim_value, psnr_value


def selection_overlay(
    composite_bgr: np.ndarray,
    selection0: np.ndarray,
    selection1: np.ndarray,
    seam_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Tint each source (green / blue) and draw the seam boundary in red."""
    vis = (composite_bgr.astype(np.float32) * 0.55).astype(np.uint8)
    tint0 = np.array([60, 200, 60], np.float32)
    tint1 = np.array([200, 90, 40], np.float32)
    vis[selection0 > 0] = (vis[selection0 > 0].astype(np.float32) * 0.4 + tint0 * 0.6).astype(np.uint8)
    vis[selection1 > 0] = (vis[selection1 > 0].astype(np.float32) * 0.4 + tint1 * 0.6).astype(np.uint8)
    boundary = cv2.morphologyEx(selection0, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    vis[boundary > 0] = seam_color
    return vis


def overlay_50(
    warped0_bgr: np.ndarray,
    warped1_bgr: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
) -> np.ndarray:
    """50% average in the overlap (double-image diagnostic)."""
    out = np.zeros_like(warped0_bgr)
    valid0 = mask0 > 0
    valid1 = mask1 > 0
    both = valid0 & valid1
    if both.any():
        out[both] = cv2.addWeighted(warped0_bgr, 0.5, warped1_bgr, 0.5, 0)[both]
    only0 = valid0 & ~both
    if only0.any():
        out[only0] = warped0_bgr[only0]
    only1 = valid1 & ~both
    if only1.any():
        out[only1] = warped1_bgr[only1]
    return out


def draw_matches(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inliers: np.ndarray | None = None,
    max_draw: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Side-by-side match canvas; MAGSAC inliers green, outliers orange."""
    canvas = np.concatenate([image0_rgb, image1_rgb], axis=1)[:, :, ::-1].copy()
    width0 = image0_rgb.shape[1]
    rng = np.random.default_rng(seed)
    indices = np.arange(len(points0))
    if len(indices) > max_draw:
        indices = rng.choice(indices, size=max_draw, replace=False)
    for index in indices:
        p0 = tuple(np.round(points0[index]).astype(int))
        p1 = tuple(np.round(points1[index]).astype(int))
        color = (120, 220, 120) if inliers is None else ((60, 200, 60) if inliers[index] else (40, 140, 230))
        cv2.line(canvas, p0, (p1[0] + width0, p1[1]), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, (p1[0] + width0, p1[1]), 3, color, -1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def label_image(image_bgr: np.ndarray, label: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (max(120, 10 * len(label)), 26), (20, 20, 20), -1)
    cv2.putText(out, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(
    images: list[np.ndarray],
    labels: list[str],
    *,
    height: int = 460,
    gap: int = 10,
) -> np.ndarray:
    """Resize to a common height, label and concatenate horizontally (RGB output)."""
    panels = []
    for image, label in zip(images, labels, strict=True):
        scale = height / image.shape[0]
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), height),
            interpolation=cv2.INTER_AREA,
        )
        panels.append(label_image(resized, label))
    canvas = np.full((height, sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1), 3), 24, np.uint8)
    x = 0
    for panel in panels:
        canvas[:, x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
