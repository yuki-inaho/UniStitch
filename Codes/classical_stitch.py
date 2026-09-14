"""Classical stitching baseline: shared LightGlue+ matches + MAGSAC++ + seam cut.

This is the comparison partner for UniStitch in the gradio app. It consumes the
*same* keypoint matching result as the network pipeline (no second matcher run),
estimates a homography with OpenCV's USAC_MAGSAC++ and warps both images into a
common canvas. The finish (GraphCut seam cut, one source per pixel) is applied by
the caller via ``stitch_common.seam_compose`` so both methods share it exactly.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from stitch_common import masked_ssim_psnr, overlap_mask, seam_compose, warp_with_mask


def estimate_homography(
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    reproj_threshold: float = 3.0,
    max_iters: int = 10000,
    confidence: float = 0.9999,
) -> tuple[np.ndarray, np.ndarray]:
    """MAGSAC++ homography mapping points of image1 onto image0."""
    if len(points0) < 4 or len(points1) < 4:
        raise RuntimeError(f"MAGSAC++ needs at least 4 matches (got {len(points0)})")
    homography, inliers = cv2.findHomography(
        points1.astype(np.float64),
        points0.astype(np.float64),
        cv2.USAC_MAGSAC,
        float(reproj_threshold),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if homography is None or inliers is None:
        raise RuntimeError("MAGSAC++ homography estimation failed")
    return homography, inliers.ravel().astype(bool)


def canvas_matrices(
    homography_tgt2ref: np.ndarray,
    ref_size: tuple[int, int],
    tgt_size: tuple[int, int],
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Place the reference at the bounding-box origin; return canvas + matrices."""
    height0, width0 = ref_size
    height1, width1 = tgt_size

    def corners(width: int, height: int) -> np.ndarray:
        return np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64)

    reference = corners(width0, height0)
    target = cv2.perspectiveTransform(corners(width1, height1)[None], homography_tgt2ref)[0]
    points = np.concatenate([reference, target], axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    canvas_size = (
        max(16, int(np.ceil(maximum[0] - minimum[0]))),
        max(16, int(np.ceil(maximum[1] - minimum[1]))),
    )
    translation = np.array(
        [[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]],
        np.float64,
    )
    return canvas_size, translation, translation @ homography_tgt2ref


def warp_classical_pair(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    match: dict[str, np.ndarray],
    *,
    reproj_threshold: float = 3.0,
    max_canvas_side: int = 6000,
) -> dict:
    """Align the pair with MAGSAC++ and warp both images into a common canvas.

    Returns warped BGR images, validity masks, MAGSAC inliers, overlap metrics
    and timings; the caller applies the shared seam finish.
    """
    image0_bgr = cv2.cvtColor(image0_rgb, cv2.COLOR_RGB2BGR)
    image1_bgr = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)

    started = time.time()
    homography, inliers = estimate_homography(
        match["keypoints0"],
        match["keypoints1"],
        reproj_threshold=reproj_threshold,
    )
    align_seconds = time.time() - started

    canvas_size, matrix0, matrix1 = canvas_matrices(homography, image0_bgr.shape[:2], image1_bgr.shape[:2])
    if max(canvas_size) > max_canvas_side:
        raise RuntimeError(f"stitched canvas too large ({canvas_size[0]}x{canvas_size[1]})")

    warped0, mask0 = warp_with_mask(image0_bgr, matrix0, canvas_size)
    warped1, mask1 = warp_with_mask(image1_bgr, matrix1, canvas_size)
    warp_seconds = time.time() - started - align_seconds

    overlap = overlap_mask(mask0, mask1)
    ssim, psnr = masked_ssim_psnr(warped0, warped1, overlap)

    return {
        "method": "classical",
        "flag_check": True,
        "warped_bgr": (warped0, warped1),
        "masks": (mask0, mask1),
        "inliers": inliers,
        "homography": homography,
        "canvas_size": canvas_size,
        "ssim": ssim,
        "psnr": psnr,
        "overlap_fraction": float((overlap > 0).mean()),
        "align_seconds": align_seconds,
        "warp_seconds": warp_seconds,
    }


def stitch_classical(
    image0_rgb: np.ndarray,
    image1_rgb: np.ndarray,
    match: dict[str, np.ndarray],
    *,
    reproj_threshold: float = 3.0,
    max_canvas_side: int = 6000,
) -> dict:
    """Full classical result (alignment + shared seam cut), convenient for tests."""
    result = warp_classical_pair(
        image0_rgb,
        image1_rgb,
        match,
        reproj_threshold=reproj_threshold,
        max_canvas_side=max_canvas_side,
    )
    warped0, warped1 = result["warped_bgr"]
    mask0, mask1 = result["masks"]
    started = time.time()
    composite, selection0, selection1 = seam_compose(warped0, warped1, mask0, mask1)
    result.update(
        {
            "composite_bgr": composite,
            "composite_rgb": cv2.cvtColor(composite, cv2.COLOR_BGR2RGB),
            "selection": (selection0, selection1),
            "seam_seconds": time.time() - started,
        }
    )
    result["total_seconds"] = result["align_seconds"] + result["warp_seconds"] + result["seam_seconds"]
    return result
