#!/usr/bin/env python3
"""Build a UniStitch dataset (input1/input2) from a sequential capture session.

Pairs are ``(frame_i, frame_{i+stride})`` from a session directory containing
``Color_*.jpg`` frames (e.g. the L76 tomato captures). Images are optionally
rotated (the capture camera is mounted sideways, so ``--rotate ccw90`` makes
them upright) and written as

    input1/<tag>_<i:06d>_<j:06d>.jpg   (frame i)
    input2/<tag>_<i:06d>_<j:06d>.jpg   (frame j)

so that ``Codes/dataset.py`` pairs them by file name and the keypoint tool can
build an ``aliked_lmdb`` next to them.

Example::

    pixi run python tools/build_sequential_pairs.py \
        --session /workspace/data/.../NYX650_2026_06_30_17_46_20_1998 \
        --output data/l76_s0 --tag s0 --stride 50 --rotate ccw90
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path

import cv2
from beartype import beartype

ROTATIONS = {
    "none": None,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
}


@beartype
def session_frames(session: Path) -> list[Path]:
    frames = sorted(session.glob("Color_*.jpg"))
    if not frames:
        frames = sorted(session.glob("Color_*.png"))
    if not frames:
        raise SystemExit(f"no Color_* frames found in {session}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, required=True, help="session dir containing Color_*.jpg")
    parser.add_argument("--output", type=Path, required=True, help="dataset dir (gets input1/ and input2/)")
    parser.add_argument("--tag", type=str, default="s0")
    parser.add_argument("--stride", type=int, default=50, help="frame gap between the pair members")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of pairs")
    parser.add_argument("--rotate", choices=sorted(ROTATIONS), default="ccw90")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    frames = session_frames(args.session)
    pairs = [(index, index + args.stride) for index in range(len(frames) - args.stride)]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"{args.session.name}: {len(frames)} frames -> {len(pairs)} pairs (stride {args.stride})")

    input1_dir = args.output / "input1"
    input2_dir = args.output / "input2"
    input1_dir.mkdir(parents=True, exist_ok=True)
    input2_dir.mkdir(parents=True, exist_ok=True)
    rotation = ROTATIONS[args.rotate]

    def write_pair(pair: tuple[int, int]) -> None:
        index, other = pair
        name = f"{args.tag}_{index:06d}_{other:06d}.jpg"
        for frame_index, directory in ((index, input1_dir), (other, input2_dir)):
            image = cv2.imread(str(frames[frame_index]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read {frames[frame_index]}")
            if rotation is not None:
                image = cv2.rotate(image, rotation)
            if not cv2.imwrite(str(directory / name), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
                raise RuntimeError(f"failed to write {directory / name}")

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        list(executor.map(write_pair, pairs))
    print(f"wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
