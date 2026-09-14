#!/usr/bin/env python3
"""RANSAC matching statistics and pose estimation vs. ground-truth poses.

Reads matched keypoints from a keypoint lmdb (built by
``keypoint_tool/onnx_keypoint_tool.py``), estimates homographies and relative
poses with RANSAC, and compares them against the capture's refined poses
(``poses_vggt/lane_refined.npz``). Also reports signals about whether the GT
poses behave like ground truth (rail consistency, per-step distances and
rotations, chunk scales).

Example::

    pixi run python tools/analyze_matches_poses.py \
        --lmdb data/l76_s0/aliked_lmdb --session /workspace/data/.../NYX650_2026_06_30_17_46_20_1998 \
        --stride 50 --pairs 200 --output /tmp/l76_pose_stats.json

    # matching statistics only (no GT), e.g. for UDIS-D
    pixi run python tools/analyze_matches_poses.py --lmdb data/UDIS-D/testing/aliked_lmdb --pairs 100
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import yaml
from beartype import beartype
from jaxtyping import Float

ROTATIONS = {"none": None, "ccw90": "ccw90", "cw90": "cw90"}


@beartype
def read_intrinsics(session: Path) -> Float[np.ndarray, "3 3"]:
    parameters = yaml.safe_load((session / "standard/camera_parameters/rgb_camera_param.yaml").read_text())
    return np.array(parameters["K"], dtype=np.float64).reshape(3, 3)


@beartype
def read_poses(session: Path) -> tuple[list[str], Float[np.ndarray, "n 4 4"], dict[str, Any]]:
    poses = np.load(session / "poses_vggt/lane_refined.npz")
    stems = [str(stem) for stem in poses["frame_stems"]]
    extra = {}
    for key in ("rail_inlier_mask", "chunk_scales", "rail_centroid", "rail_axis", "projection_strength"):
        if key in poses:
            extra[key] = poses[key]
    return stems, poses["camera_to_global"], extra


@beartype
def lmdb_pair(lmdb_path: Path, index: int) -> tuple[Float[np.ndarray, "n 2"], Float[np.ndarray, "n 2"]] | None:
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn, txn.cursor() as cursor:
            if not cursor.set_key(f"{index:08d}".encode()):
                cursor.first()
                reached = False
                for position, _ in enumerate(cursor):
                    if position == index:
                        reached = True
                        break
                    cursor.next()
                if not reached:
                    return None
            data = pickle.loads(cursor.value())
    finally:
        env.close()
    return (
        np.asarray(data["keypoints0"], dtype=np.float32),
        np.asarray(data["keypoints1"], dtype=np.float32),
    )


@beartype
def lmdb_entries(lmdb_path: Path, limit: int | None) -> list[tuple[bytes, dict[str, np.ndarray]]]:
    entries: list[tuple[bytes, dict[str, np.ndarray]]] = []
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn, txn.cursor() as cursor:
            for index, (key, value) in enumerate(cursor):
                entries.append((bytes(key), pickle.loads(value)))
                if limit is not None and index + 1 >= limit:
                    break
    finally:
        env.close()
    return entries


@beartype
def original_frame_points(
    points: Float[np.ndarray, "n 2"], original_width: int, rotation: str
) -> Float[np.ndarray, "n 2"]:
    """Map keypoints from the rotated image back to the original camera frame."""
    if rotation == "none":
        return points
    if rotation == "ccw90":
        # cv2.ROTATE_90_COUNTERCLOCKWISE: (x_o, y_o) -> (y_o, W-1-x_o)
        x = original_width - 1.0 - points[:, 1]
        y = points[:, 0]
        return np.stack([x, y], axis=1)
    if rotation == "cw90":
        # cv2.ROTATE_90_CLOCKWISE: (x_o, y_o) -> (H-1-y_o, x_o) -> inverse:
        raise ValueError("cw90 inverse needs the original height; use --rotate ccw90 or none")
    raise ValueError(f"unknown rotation {rotation}")


@beartype
def rotation_angle_deg(rotation: Float[np.ndarray, "3 3"]) -> float:
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


@beartype
def vector_angle_deg(a: Float[np.ndarray, "3"], b: Float[np.ndarray, "3"]) -> float:
    numerator = float(np.dot(a, b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cosine = max(-1.0, min(1.0, numerator / denominator))
    return float(math.degrees(math.acos(cosine)))


@beartype
def analyze_pairs(
    lmdb_path: Path,
    *,
    pairs: int | None,
    ransac_px: float,
    session: Path | None,
    stride: int,
    rotation: str,
    original_width: int,
    ground_truth: bool,
) -> dict[str, Any]:
    entries = lmdb_entries(lmdb_path, pairs)
    intrinsic = read_intrinsics(session) if ground_truth and session is not None else None
    if ground_truth and session is not None:
        assert session is not None
        stems, camera_to_global, extra = read_poses(session)
        stem_to_index = {stem: index for index, stem in enumerate(stems)}
    else:
        camera_to_global, extra, stem_to_index = None, {}, {}

    rows = []
    for index, (_key, data) in enumerate(entries):
        points0 = np.asarray(data["keypoints0"], dtype=np.float32)
        points1 = np.asarray(data["keypoints1"], dtype=np.float32)
        points0 = original_frame_points(points0, original_width, rotation)
        points1 = original_frame_points(points1, original_width, rotation)
        row: dict[str, Any] = {"index": index, "matches": int(len(points0))}
        if len(points0) >= 8:
            homography, mask = cv2.findHomography(points0, points1, cv2.RANSAC, ransac_px)
            if homography is not None and mask is not None:
                projected = cv2.perspectiveTransform(points0[None], homography)[0]
                errors = np.linalg.norm(projected - points1, axis=1)
                inliers = mask.ravel().astype(bool)
                row.update(
                    h_inliers=int(inliers.sum()),
                    h_inlier_ratio=float(inliers.mean()),
                    h_reproj_mean=float(errors[inliers].mean()) if inliers.any() else None,
                )
        if intrinsic is not None and len(points0) >= 8:
            assert camera_to_global is not None and session is not None
            essential, mask = cv2.findEssentialMat(points0, points1, intrinsic, cv2.RANSAC, 0.999, 1.0)
            if essential is not None and mask is not None and essential.shape == (3, 3):
                _, rotation_est, translation_est, _ = cv2.recoverPose(essential, points0, points1, intrinsic)
                stem = f"{index + 1:08d}"
                stem_other = f"{index + 1 + stride:08d}"
                if stem in stem_to_index and stem_other in stem_to_index:
                    first = camera_to_global[stem_to_index[stem]]
                    second = camera_to_global[stem_to_index[stem_other]]
                    relative = np.linalg.inv(second) @ first
                    rotation_gt = relative[:3, :3]
                    translation_gt = relative[:3, 3]
                    translation_est = translation_est.ravel()
                    row.update(
                        e_inliers=int(mask.sum()),
                        e_inlier_ratio=float(mask.mean()),
                        rot_est_gt_deg=rotation_angle_deg(rotation_est.T @ rotation_gt),
                        trans_est_gt_deg=min(
                            vector_angle_deg(translation_est, translation_gt),
                            vector_angle_deg(translation_est, -translation_gt),
                        ),
                        gt_baseline_m=float(np.linalg.norm(translation_gt)),
                        gt_rotation_deg=rotation_angle_deg(rotation_gt),
                    )
        rows.append(row)

    def stats(key: str) -> dict[str, float] | None:
        values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
        if not values.size:
            return None
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary: dict[str, Any] = {
        "lmdb": str(lmdb_path),
        "pairs": len(rows),
        "matches": stats("matches"),
        "h_inlier_ratio": stats("h_inlier_ratio"),
        "h_inliers": stats("h_inliers"),
        "h_reproj_mean": stats("h_reproj_mean"),
        "e_inlier_ratio": stats("e_inlier_ratio"),
        "rot_est_gt_deg": stats("rot_est_gt_deg"),
        "trans_est_gt_deg": stats("trans_est_gt_deg"),
        "gt_baseline_m": stats("gt_baseline_m"),
        "gt_rotation_deg": stats("gt_rotation_deg"),
        "rows": rows[:50],
    }
    if ground_truth and camera_to_global is not None:
        steps = np.linalg.norm(np.diff(camera_to_global[:, :3, 3], axis=0), axis=1)
        angles = np.array(
            [
                rotation_angle_deg(camera_to_global[i, :3, :3].T @ camera_to_global[i + 1, :3, :3])
                for i in range(len(camera_to_global) - 1)
            ]
        )
        summary["gt_sanity"] = {
            "step_m_mean": float(steps.mean()),
            "step_m_median": float(np.median(steps)),
            "step_m_p95": float(np.percentile(steps, 95)),
            "step_rot_deg_mean": float(angles.mean()),
            "step_rot_deg_p95": float(np.percentile(angles, 95)),
            "rail_inlier_fraction": float(np.mean(extra["rail_inlier_mask"])) if "rail_inlier_mask" in extra else None,
            "chunk_scale_mean": float(np.mean(extra["chunk_scales"])) if "chunk_scales" in extra else None,
            "chunk_scale_cv": float(np.std(extra["chunk_scales"]) / (np.mean(extra["chunk_scales"]) + 1e-12))
            if "chunk_scales" in extra
            else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=100, help="number of lmdb entries to analyze (null = all)")
    parser.add_argument("--ransac_px", type=float, default=3.0)
    parser.add_argument("--session", type=Path, default=None, help="capture session for GT poses (optional)")
    parser.add_argument("--stride", type=int, default=50, help="frame gap used when the dataset was built")
    parser.add_argument("--rotate", choices=sorted(ROTATIONS), default="ccw90")
    parser.add_argument("--original_width", type=int, default=800)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze_pairs(
        args.lmdb,
        pairs=args.pairs,
        ransac_px=args.ransac_px,
        session=args.session,
        stride=args.stride,
        rotation=args.rotate,
        original_width=args.original_width,
        ground_truth=args.session is not None,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
