#!/usr/bin/env python3
"""Build self-contained keypoint databases (lmdb) for UniStitch datasets.

Two ONNX backends are available (see ``onnx_models.py``):

  * ``raco``   - RaCo-ALIKED-LightGlue+ (LightGlue-ONNX v3.0, default)
  * ``aliked`` - COLMAP 3.13 ALIKED + ALIKED-LightGlue

A dataset folder must contain ``input1/`` and ``input2/`` (see
``Codes/dataset.py``). The database is written to ``<data_path>/<name>_lmdb``
with one entry per pair under the key ``{index:08d}_{image_name}``, holding a
pickled dict with ``keypoints0/1`` (matched points, original pixel coordinates)
and ``descriptors0/1``.

Examples::

    pixi run python keypoint_tool/onnx_keypoint_tool.py probe
    pixi run python keypoint_tool/onnx_keypoint_tool.py build --data_path data/UDIS-D/training --workers 32
    pixi run python keypoint_tool/onnx_keypoint_tool.py build --backend aliked --data_path ... --limit 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import os
import pickle
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
from beartype import beartype
from jaxtyping import UInt8
from onnx_models import (
    BACKENDS,
    DEFAULT_ALIKED_DESCRIPTOR,
    DEFAULT_ALIKED_LIGHTGLUE,
    DEFAULT_ALIKED_N16ROT,
    DEFAULT_RACO_PIPELINE,
    AlikedLightGlue,
    OnnxSession,
    RaCoAlikedLightGlue,
)

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")

_worker_state = threading.local()


@beartype
def _read_rgb(path: Path) -> UInt8[np.ndarray, "h w 3"]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@beartype
def _list_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(folder.glob(extension))
    return sorted(set(images))


@beartype
def _pair_images(input1_dir: Path, input2_dir: Path) -> list[tuple[Path, Path, str]]:
    """Pair images by file name, falling back to sorted order."""
    images1 = _list_images(input1_dir)
    images2 = _list_images(input2_dir)
    by_name = {path.name: path for path in images2}
    pairs: list[tuple[Path, Path, str]] = []
    for index, path1 in enumerate(images1):
        path2 = by_name.get(path1.name)
        if path2 is None:
            if index >= len(images2):
                raise RuntimeError(f"cannot pair {path1.name}: input2 has no matching name and no fallback entry")
            path2 = images2[index]
        pairs.append((path1, path2, path1.name))
    return pairs


def _make_backend(args: argparse.Namespace):
    if args.backend == "raco":
        return RaCoAlikedLightGlue(
            pipeline_model=args.raco_pipeline,
            descriptor_model=args.aliked_descriptor,
            matcher_min_score=args.matcher_min_score,
            num_threads=args.ort_threads,
        )
    return AlikedLightGlue(
        extractor_model=args.aliked_model,
        matcher_model=args.aliked_lightglue,
        max_keypoints=args.max_keypoints,
        min_score=args.min_score,
        matcher_min_score=args.matcher_min_score,
        num_threads=args.ort_threads,
    )


def _get_backend(args: argparse.Namespace):
    """Lazily create one backend per worker thread."""
    backend = getattr(_worker_state, "backend", None)
    if backend is None:
        backend = _make_backend(args)
        _worker_state.backend = backend
    return backend


def build_lmdb(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path).resolve()
    input1_dir = data_path / "input1"
    input2_dir = data_path / "input2"
    if not input1_dir.is_dir() or not input2_dir.is_dir():
        raise SystemExit(f"{data_path} must contain input1/ and input2/")

    lmdb_path = Path(args.lmdb_path).resolve() if args.lmdb_path else data_path / f"{args.keypoint}_lmdb"
    pairs = _pair_images(input1_dir, input2_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"[build] backend={args.backend} pairs={len(pairs)} data={data_path}")
    print(f"[build] lmdb={lmdb_path}")

    if lmdb_path.exists() and args.overwrite:
        shutil.rmtree(lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_path), map_size=int(args.map_size_gb * (1024**3)), readonly=False, readahead=False)
    existing: set[bytes] = set()
    if not args.overwrite:
        with env.begin(write=False) as txn, txn.cursor() as cursor:
            existing = {bytes(key) for key, _ in cursor}
        if existing:
            print(f"[build] resuming: {len(existing)} entries already present")

    def worker(pair: tuple[Path, Path, str]) -> dict:
        path1, path2, _name = pair
        backend = _get_backend(args)
        image0 = _read_rgb(path1)
        image1 = _read_rgb(path2)
        if args.backend == "raco":
            return backend.match_pair(image0, image1, max_side=args.max_side)
        return backend.match_pair(image0, image1)

    started = time.time()
    written = 0
    commit_every = max(1, args.commit_every)
    pending = 0
    txn = env.begin(write=True)
    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {}
            for index, pair in enumerate(pairs):
                key = f"{index:08d}_{pair[2]}".encode()
                if key in existing:
                    continue
                futures[executor.submit(worker, pair)] = (index, key)

            for count, future in enumerate(cf.as_completed(futures), 1):
                index, key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # keep going and report the failure
                    print(f"[build] ERROR index={index}: {exc}", file=sys.stderr)
                    continue
                txn.put(key, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
                written += 1
                pending += 1
                if pending >= commit_every:
                    txn.commit()
                    txn = env.begin(write=True)
                    pending = 0
                if count % args.log_every == 0 or count == len(futures):
                    elapsed = time.time() - started
                    rate = count / elapsed if elapsed > 0 else 0.0
                    remaining = (len(futures) - count) / rate if rate > 0 else 0.0
                    print(f"[build] {count}/{len(futures)} ({rate:.2f} pairs/s, ETA {remaining / 60:.1f} min)")
        txn.commit()
        txn = env.begin(write=True)
        pending = 0
    finally:
        with contextlib.suppress(Exception):
            txn.abort()
        env.close()
    print(f"[build] done: {written} new entries in {lmdb_path}")


def cmd_probe(args: argparse.Namespace) -> None:
    models = {
        "raco pipeline": args.raco_pipeline,
        "aliked descriptor": args.aliked_descriptor,
        "aliked extractor": args.aliked_model,
        "aliked lightglue": args.aliked_lightglue,
    }
    for label, model_path in models.items():
        path = Path(model_path)
        print(f"== {label}: {path}")
        if not path.exists():
            print("   MISSING")
            continue
        print(f"   sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
        session = OnnxSession(path)
        print(f"   inputs : {[(entry.name, entry.shape, entry.type) for entry in session.session.get_inputs()]}")
        print(f"   outputs: {[(entry.name, entry.shape, entry.type) for entry in session.session.get_outputs()]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=sorted(BACKENDS), default="raco")
    common.add_argument("--raco_pipeline", type=Path, default=DEFAULT_RACO_PIPELINE)
    common.add_argument("--aliked_descriptor", type=Path, default=DEFAULT_ALIKED_DESCRIPTOR)
    common.add_argument("--aliked_model", type=Path, default=DEFAULT_ALIKED_N16ROT)
    common.add_argument("--aliked_lightglue", type=Path, default=DEFAULT_ALIKED_LIGHTGLUE)
    common.add_argument("--ort_threads", type=int, default=None, help="ONNX Runtime intra-op threads per worker")

    raco = argparse.ArgumentParser(add_help=False)
    raco.add_argument("--max_side", type=int, default=1024, help="RaCo backend: longest side after resizing")
    raco.add_argument("--matcher_min_score", type=float, default=0.1, help="LightGlue match score threshold")

    aliked = argparse.ArgumentParser(add_help=False)
    aliked.add_argument("--max_keypoints", type=int, default=2048)
    aliked.add_argument("--min_score", type=float, default=0.2, help="ALIKED detection score threshold")

    build = subparsers.add_parser("build", parents=[common, raco, aliked], help="build a keypoint lmdb")
    build.add_argument("--data_path", type=Path, required=True, help="dataset folder containing input1/ and input2/")
    build.add_argument("--lmdb_path", type=Path, default=None)
    build.add_argument("--keypoint", type=str, default="aliked", help="default lmdb folder name (<keypoint>_lmdb)")
    build.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 8))
    build.add_argument("--map_size_gb", type=float, default=32.0)
    build.add_argument("--limit", type=int, default=None, help="process only the first N pairs (debugging)")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--commit_every", type=int, default=200, help="lmdb commit interval for resumable runs")
    build.add_argument("--log_every", type=int, default=25)
    build.set_defaults(func=build_lmdb)

    probe = subparsers.add_parser("probe", parents=[common], help="print model I/O signatures and checksums")
    probe.set_defaults(func=cmd_probe)

    args = parser.parse_args()
    if args.ort_threads is None and args.command == "build":
        args.ort_threads = max(1, 8 // max(1, args.workers)) if args.workers > 1 else None
    args.func(args)


if __name__ == "__main__":
    main()
