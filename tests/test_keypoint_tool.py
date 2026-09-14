"""Fast smoke tests for the self-contained ONNX keypoint pipeline.

Run with ``pixi run test`` (uses the vendored models, no network access).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "keypoint_tool" / "models"
if str(ROOT / "keypoint_tool") not in sys.path:
    sys.path.insert(0, str(ROOT / "keypoint_tool"))

from onnx_keypoint_tool import build_lmdb  # noqa: E402
from onnx_models import (  # noqa: E402
    DEFAULT_ALIKED_DESCRIPTOR,
    DEFAULT_ALIKED_LIGHTGLUE,
    DEFAULT_ALIKED_N16ROT,
    DEFAULT_RACO_PIPELINE,
    AlikedLightGlue,
    RaCoAlikedLightGlue,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model_checksums() -> None:
    """Vendored ONNX models match the pinned SHA256SUMS."""
    lines = (MODELS / "SHA256SUMS").read_text().splitlines()
    assert lines, "SHA256SUMS is empty"
    for line in lines:
        digest, name = line.split()
        path = MODELS / name
        assert path.exists(), f"missing vendored model: {name}"
        assert _sha256(path) == digest, f"checksum mismatch for {name}"


@pytest.fixture(scope="module")
def image_pair() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic textured BGR pair with a known translation (plenty of matches)."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.normal(128, 60, (400, 500, 3)).astype(np.float32), (0, 0), 4)
    for _ in range(30):
        start = tuple(int(value) for value in rng.integers(20, 380, size=2))
        end = tuple(int(value) for value in rng.integers(20, 470, size=2))
        color = tuple(float(value) for value in rng.integers(0, 255, size=3))
        cv2.line(base, (end[1], end[0]), (start[1], start[0]), color, int(rng.integers(1, 3)))
    image0 = np.clip(base, 0, 255).astype(np.uint8)
    image1 = np.roll(image0, shift=(12, 18), axis=(0, 1))
    return image0, image1


@pytest.fixture(scope="module")
def raco_backend() -> RaCoAlikedLightGlue:
    return RaCoAlikedLightGlue(num_threads=4)


def test_raco_backend_matches(image_pair: tuple[np.ndarray, np.ndarray], raco_backend: RaCoAlikedLightGlue) -> None:
    image0, image1 = image_pair
    result = raco_backend.match_pair(image0, image1, max_side=384)
    assert result["keypoints0"].shape == result["descriptors0"].shape[:1] + (2,)
    assert result["descriptors0"].shape[1] == 128
    assert result["keypoints0"].shape[0] > 0, "expected at least one match"
    norms = np.linalg.norm(result["descriptors0"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    assert result["keypoints0"][:, 0].max() <= image0.shape[1]
    assert result["keypoints0"][:, 1].max() <= image0.shape[0]


def test_colmap_aliked_backend(image_pair: tuple[np.ndarray, np.ndarray]) -> None:
    backend = AlikedLightGlue(num_threads=4)
    image0, image1 = image_pair
    result = backend.match_pair(image0, image1)
    assert result["descriptors0"].shape[1] == 128
    assert result["keypoints0"].shape[0] > 0


def test_build_lmdb_roundtrip(tmp_path: Path, image_pair: tuple[np.ndarray, np.ndarray]) -> None:
    data_path = tmp_path / "dataset"
    for folder, image in (("input1", image_pair[0]), ("input2", image_pair[1])):
        (data_path / folder).mkdir(parents=True)
        assert cv2.imwrite(str(data_path / folder / "000001.png"), image)

    args = argparse.Namespace(
        backend="raco",
        raco_pipeline=DEFAULT_RACO_PIPELINE,
        aliked_descriptor=DEFAULT_ALIKED_DESCRIPTOR,
        aliked_model=DEFAULT_ALIKED_N16ROT,
        aliked_lightglue=DEFAULT_ALIKED_LIGHTGLUE,
        matcher_min_score=0.1,
        max_keypoints=2048,
        min_score=0.2,
        max_side=384,
        data_path=data_path,
        lmdb_path=None,
        keypoint="aliked",
        workers=1,
        map_size_gb=1.0,
        limit=1,
        overwrite=True,
        commit_every=10,
        log_every=10,
        ort_threads=4,
    )
    build_lmdb(args)

    env = lmdb.open(str(data_path / "aliked_lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.stat()["entries"] == 1
        value = txn.get(b"00000000_000001.png")
        assert value is not None
        data = pickle.loads(value)
    assert set(data) == {"keypoints0", "keypoints1", "descriptors0", "descriptors1"}
    assert data["keypoints0"].shape[1] == 2
    assert data["descriptors0"].shape[1] == 128
