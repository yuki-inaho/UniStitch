#!/usr/bin/env python3
"""Select dataset pairs for training, e.g. by near-view (close-range) content.

Scores every pair in a keypoint lmdb with a criterion and materializes a subset
dataset (symlinked images + a re-indexed keypoint lmdb) so it can be fed
directly to ``Codes/train.py``.

Criteria:

``near_fraction``
    Fraction of valid depth pixels closer than ``--near_mm`` (the L76 capture
    has aligned depth maps under ``standard/mapped_depth``). Useful to
    emphasise close-range scenes.

``homography_inliers``
    RANSAC homography inlier ratio of the matches (prefer planar/parallax-free
    pairs; low values indicate parallax).

Example::

    pixi run python tools/select_pairs.py \\
        --source data/l76_s0 --output data/l76_s0_near --criterion near_fraction \\
        --depth_dir /workspace/data/.../standard/mapped_depth \\
        --stride 50 --keep_fraction 0.4
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
from beartype import beartype
from jaxtyping import Float

CRITERIA = ("near_fraction", "homography_inliers")


@beartype
def frame_near_fractions(depth_dir: Path, near_mm: int, frames: int) -> Float[np.ndarray, "n"]:
    """Per-frame near-content fraction (valid pixels closer than ``near_mm``)."""
    scores = np.zeros(frames, dtype=np.float64)
    for index in range(frames):
        path = depth_dir / f"{index + 1:08d}_depth.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        valid = image > 0
        if not valid.any():
            continue
        scores[index] = float(np.mean(valid & (image < near_mm)))
    return scores


@beartype
def load_entries(lmdb_path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    entries: list[tuple[bytes, dict[str, Any]]] = []
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn, txn.cursor() as cursor:
            for key, value in cursor:
                entries.append((bytes(key), pickle.loads(value)))
    finally:
        env.close()
    return entries


@beartype
def homography_scores(
    entries: list[tuple[bytes, dict[str, Any]]], original_width: int, ransac_px: float
) -> Float[np.ndarray, "n"]:
    scores = np.zeros(len(entries), dtype=np.float64)
    for index, (_key, data) in enumerate(entries):
        points0 = np.asarray(data["keypoints0"], dtype=np.float32)
        points1 = np.asarray(data["keypoints1"], dtype=np.float32)
        x = original_width - 1.0 - points0[:, 1]
        points0 = np.stack([x, points0[:, 0]], axis=1)
        x = original_width - 1.0 - points1[:, 1]
        points1 = np.stack([x, points1[:, 0]], axis=1)
        if len(points0) < 8:
            continue
        _homography, mask = cv2.findHomography(points0, points1, cv2.RANSAC, ransac_px)
        if mask is not None:
            scores[index] = float(mask.mean())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="source dataset dir (input1/input2 + aliked_lmdb)")
    parser.add_argument("--output", type=Path, required=True, help="output dataset dir")
    parser.add_argument("--criterion", choices=CRITERIA, default="near_fraction")
    parser.add_argument("--lmdb_name", type=str, default="aliked_lmdb")
    parser.add_argument("--depth_dir", type=Path, default=None, help="mapped depth dir for near_fraction")
    parser.add_argument("--near_mm", type=int, default=1200)
    parser.add_argument("--stride", type=int, default=50, help="frame gap used when the dataset was built")
    parser.add_argument("--original_width", type=int, default=800)
    parser.add_argument("--ransac_px", type=float, default=3.0)
    parser.add_argument("--keep_fraction", type=float, default=0.4)
    parser.add_argument(
        "--threshold", type=float, default=None, help="use an absolute score threshold instead of keep_fraction"
    )
    parser.add_argument("--keep_lowest", action="store_true", help="keep the lowest scores instead of the highest")
    args = parser.parse_args()

    source_lmdb = args.source / args.lmdb_name
    entries = load_entries(source_lmdb)

    if args.criterion == "near_fraction":
        if args.depth_dir is None:
            raise SystemExit("--depth_dir is required for near_fraction")
        frame_scores = frame_near_fractions(args.depth_dir, args.near_mm, len(entries) + args.stride)
        scores = np.array(
            [0.5 * (frame_scores[i] + frame_scores[i + args.stride]) for i in range(len(entries))],
            dtype=np.float64,
        )
        detail_key = "near_fraction"
    else:
        scores = homography_scores(entries, args.original_width, args.ransac_px)
        detail_key = "homography_inlier_ratio"

    order = np.argsort(scores)
    if not args.keep_lowest:
        order = order[::-1]
    if args.threshold is not None:
        selected = [
            int(i) for i in order if (scores[i] <= args.threshold if args.keep_lowest else scores[i] >= args.threshold)
        ]
    else:
        keep = max(1, int(round(len(order) * args.keep_fraction)))
        selected = [int(i) for i in order[:keep]]
    # Re-index in source order so the new lmdb keys match the sorted file names
    # that Codes/dataset.py enumerates.
    selected = sorted(selected)

    output_aliked = args.output / args.lmdb_name
    input1_dir = args.output / "input1"
    input2_dir = args.output / "input2"
    output_aliked.parent.mkdir(parents=True, exist_ok=True)
    input1_dir.mkdir(parents=True, exist_ok=True)
    input2_dir.mkdir(parents=True, exist_ok=True)
    for directory in (input1_dir, input2_dir):
        for existing in directory.glob("*.jpg"):
            existing.unlink()
    if output_aliked.exists():
        shutil.rmtree(output_aliked)
    output_aliked.mkdir(parents=True)

    env = lmdb.open(str(output_aliked), map_size=8 * (1024**3), readonly=False, readahead=False)
    selection = []
    with env.begin(write=True) as txn:
        for new_index, old_index in enumerate(selected):
            key, data = entries[old_index]
            name = key.decode().split("_", 1)[1]
            txn.put(f"{new_index:08d}_{name}".encode(), pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            selection.append(
                {
                    "index": new_index,
                    "source_index": int(old_index),
                    "name": name,
                    detail_key: float(scores[old_index]),
                }
            )
            for folder in ("input1", "input2"):
                destination = args.output / folder / name
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                destination.symlink_to((args.source / folder / name).resolve())
    env.close()

    (args.output / "selection.json").write_text(
        json.dumps({"criterion": args.criterion, "selected": selection}, indent=2)
    )
    print(
        f"selected {len(selected)}/{len(entries)} pairs by {args.criterion} -> {args.output} "
        f"(score mean {scores[selected].mean():.4f})"
    )


if __name__ == "__main__":
    main()
