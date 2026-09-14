"""ONNX inference modules for keypoint extraction and matching.

Three self-contained backends are provided (all model files are vendored under
``keypoint_tool/models``):

``raco``
    RaCo-ALIKED-LightGlue+ (LightGlue-ONNX v3.0). The released pipeline detects
    keypoints and matches one interleaved image pair; a companion descriptor
    model samples the ALIKED 128-d descriptors at those keypoints. This is the
    default backend and the one used for UniStitch training data.

``aliked``
    COLMAP 3.13 ALIKED extractor + ALIKED-LightGlue matcher, following
    ``src/colmap/feature/aliked.cc`` and ``onnx_matchers.cc``.

The modules only require ``numpy``, ``opencv-python`` and ``onnxruntime``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort
from beartype import beartype
from jaxtyping import Float, Int, Shaped, UInt8

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

DEFAULT_RACO_PIPELINE = MODELS_DIR / "raco_aliked_lightglue_pipeline_k2048.onnx"
DEFAULT_ALIKED_DESCRIPTOR = MODELS_DIR / "aliked-descriptor.onnx"
DEFAULT_ALIKED_N16ROT = MODELS_DIR / "aliked-n16rot.onnx"
DEFAULT_ALIKED_LIGHTGLUE = MODELS_DIR / "aliked-lightglue.onnx"

#: COLMAP pads ALIKED inputs to a multiple of 32; the RaCo pipeline uses the same divisor.
PAD_DIVISOR = 32

#: The v3.0 pipeline is exported with dynamic spatial dims of at least 2 * divisor.
MIN_SIDE = 2 * PAD_DIVISOR


class OnnxSession:
    """Thin wrapper around ``onnxruntime.InferenceSession``."""

    @beartype
    def __init__(
        self,
        model_path: os.PathLike | str,
        *,
        providers: list[str] | None = None,
        num_threads: int | None = None,
        optimization_level: int | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel(
            optimization_level if optimization_level is not None else ort.GraphOptimizationLevel.ORT_ENABLE_ALL.value
        )
        if num_threads is not None:
            options.intra_op_num_threads = int(num_threads)
        if providers is None:
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        self.model_path = path
        self.session = ort.InferenceSession(str(path), options, providers=list(providers))
        self.input_names = [entry.name for entry in self.session.get_inputs()]
        self.output_names = [entry.name for entry in self.session.get_outputs()]

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        outputs = self.session.run(
            self.output_names, {key: value for key, value in feed.items() if key in self.input_names}
        )
        return cast(list[np.ndarray], outputs)


@dataclass
class Features:
    """Detected keypoints with their descriptors and scores."""

    points: Float[np.ndarray, "n 2"]  # pixel coordinates (x, y)
    descriptors: Float[np.ndarray, "n d"]
    scores: Float[np.ndarray, "n"]


@beartype
def _resize_to_divisor(width: int, height: int, *, max_side: int | None) -> tuple[int, int]:
    """Scale a size down so the long side is at most ``max_side``, rounding to multiples of 32."""
    scale = 1.0
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
    target_width = max(MIN_SIDE, int(round(width * scale / PAD_DIVISOR)) * PAD_DIVISOR)
    target_height = max(MIN_SIDE, int(round(height * scale / PAD_DIVISOR)) * PAD_DIVISOR)
    return target_width, target_height


class RaCoAlikedLightGlue:
    """RaCo-ALIKED-LightGlue+ pipeline (v3.0) with ALIKED descriptor sampling."""

    @beartype
    def __init__(
        self,
        pipeline_model: os.PathLike | str = DEFAULT_RACO_PIPELINE,
        descriptor_model: os.PathLike | str = DEFAULT_ALIKED_DESCRIPTOR,
        *,
        matcher_min_score: float = 0.1,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.pipeline = OnnxSession(pipeline_model, providers=providers, num_threads=num_threads)
        self.descriptor = OnnxSession(descriptor_model, providers=providers, num_threads=num_threads)
        self.matcher_min_score = float(matcher_min_score)

    @beartype
    def _blob(self, images: list[UInt8[np.ndarray, "h w 3"]], size: tuple[int, int]) -> Float[np.ndarray, "two 3 h w"]:
        return cv2.dnn.blobFromImages(
            images,
            scalefactor=1.0 / 255.0,
            size=size,
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_32F,
        ).astype(np.float32)

    @beartype
    def match_pair(
        self,
        image0: UInt8[np.ndarray, "h0 w0 3"],
        image1: UInt8[np.ndarray, "h1 w1 3"],
        *,
        max_side: int | None = 1024,
    ) -> dict[str, Float[np.ndarray, "m k"]]:
        """Match an RGB image pair, returning matched keypoints/descriptors in original pixels."""
        height0, width0 = image0.shape[:2]
        height1, width1 = image1.shape[:2]
        # Both images share one tensor, so a common working size is required. The
        # first image defines the aspect ratio (matching TestDataset's behaviour).
        target_width, target_height = _resize_to_divisor(width0, height0, max_side=max_side)
        resized0 = cv2.resize(image0, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized1 = cv2.resize(image1, (target_width, target_height), interpolation=cv2.INTER_AREA)
        images = self._blob([resized0, resized1], (target_width, target_height))

        (keypoints, matches, mscores) = self.pipeline.run({"images": images})
        keypoints = keypoints.astype(np.float32)
        matches = matches.astype(np.int64)
        mscores = mscores.astype(np.float32)

        if len(matches) and self.matcher_min_score > 0.0:
            keep = mscores >= self.matcher_min_score
            matches = matches[keep]

        (descriptors,) = self.descriptor.run({"images": images, "keypoints": keypoints})
        descriptors = descriptors.astype(np.float32)

        indices0 = matches[:, 1]
        indices1 = matches[:, 2]
        scale_x = width0 / target_width
        scale_y = height0 / target_height
        scale_x1 = width1 / target_width
        scale_y1 = height1 / target_height
        return {
            "keypoints0": np.stack([keypoints[0, indices0, 0] * scale_x, keypoints[0, indices0, 1] * scale_y], axis=1),
            "keypoints1": np.stack(
                [keypoints[1, indices1, 0] * scale_x1, keypoints[1, indices1, 1] * scale_y1], axis=1
            ),
            "descriptors0": descriptors[0, indices0],
            "descriptors1": descriptors[1, indices1],
        }


class AlikedExtractor:
    """ALIKED ONNX feature extractor (COLMAP ``AlikedFeatureExtractor`` semantics)."""

    @beartype
    def __init__(
        self,
        model_path: os.PathLike | str = DEFAULT_ALIKED_N16ROT,
        *,
        max_keypoints: int = 2048,
        min_score: float = 0.2,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.max_keypoints = int(max_keypoints)
        self.min_score = float(min_score)
        self.session = OnnxSession(model_path, providers=providers, num_threads=num_threads)
        self.descriptor_dim = 128

    @beartype
    def extract(self, image: UInt8[np.ndarray, "h w 3"]) -> Features:
        height, width = image.shape[:2]
        chw = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        padded_height = ((height + PAD_DIVISOR - 1) // PAD_DIVISOR) * PAD_DIVISOR
        padded_width = ((width + PAD_DIVISOR - 1) // PAD_DIVISOR) * PAD_DIVISOR
        if (padded_height, padded_width) != (height, width):
            chw = np.pad(
                chw,
                ((0, 0), (0, 0), (0, padded_height - height), (0, padded_width - width)),
                mode="edge",
            )
        chw = np.ascontiguousarray(chw)

        keypoints, descriptors, scores = self.session.run(
            {
                "image": chw,
                "max_keypoints": np.array(self.max_keypoints, dtype=np.int64),
                "min_score": np.array(self.min_score, dtype=np.float32),
            }
        )
        keypoints = keypoints[0].astype(np.float32)
        descriptors = descriptors[0].astype(np.float32)
        scores = scores[0].astype(np.float32)

        # Normalized [-1, 1] coordinates relative to the padded image -> pixels.
        scale_x = 0.5 * float(padded_width - 1)
        scale_y = 0.5 * float(padded_height - 1)
        px = (keypoints[:, 0] + 1.0) * scale_x + 0.5
        py = (keypoints[:, 1] + 1.0) * scale_y + 0.5
        keep = (scores >= self.min_score) & (px >= 0.0) & (px < width) & (py >= 0.0) & (py < height)
        return Features(
            points=np.stack([px[keep], py[keep]], axis=1).astype(np.float32),
            descriptors=descriptors[keep],
            scores=scores[keep],
        )


class LightGlueMatcher:
    """ALIKED-LightGlue ONNX matcher (COLMAP ``LightGlueONNXFeatureMatcher`` semantics)."""

    @beartype
    def __init__(
        self,
        model_path: os.PathLike | str = DEFAULT_ALIKED_LIGHTGLUE,
        *,
        min_score: float = 0.1,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.min_score = float(min_score)
        self.session = OnnxSession(model_path, providers=providers, num_threads=num_threads)

    @beartype
    def match(
        self,
        points0: Float[np.ndarray, "n0 2"],
        points1: Float[np.ndarray, "n1 2"],
        descriptors0: Float[np.ndarray, "n0 d"],
        descriptors1: Float[np.ndarray, "n1 d"],
        size0: tuple[int, int],
        size1: tuple[int, int],
    ) -> tuple[Int[np.ndarray, "m"], Int[np.ndarray, "m"], Float[np.ndarray, "m"]]:
        def as_batch(array: np.ndarray, dtype: np.dtype | type = np.float32) -> np.ndarray:
            return np.ascontiguousarray(array[None].astype(dtype))

        feed = {
            "kpts0": as_batch(points0 - 0.5),
            "kpts1": as_batch(points1 - 0.5),
            "desc0": as_batch(descriptors0),
            "desc1": as_batch(descriptors1),
            "image_size0": np.array([[size0[0], size0[1]]], dtype=np.float32),
            "image_size1": np.array([[size1[0], size1[1]]], dtype=np.float32),
        }
        matches0, mscores0 = self.session.run(feed)
        matches0 = np.asarray(matches0).reshape(-1)
        mscores0 = np.asarray(mscores0).reshape(-1)
        keep = (matches0 >= 0) & (mscores0 >= self.min_score)
        indices0 = np.nonzero(keep)[0].astype(np.int64)
        indices1 = matches0[indices0].astype(np.int64)
        return indices0, indices1, mscores0[indices0]


class AlikedLightGlue:
    """COLMAP-style ALIKED extraction + ALIKED-LightGlue matching."""

    @beartype
    def __init__(
        self,
        *,
        extractor_model: os.PathLike | str = DEFAULT_ALIKED_N16ROT,
        matcher_model: os.PathLike | str = DEFAULT_ALIKED_LIGHTGLUE,
        max_keypoints: int = 2048,
        min_score: float = 0.2,
        matcher_min_score: float = 0.1,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.extractor = AlikedExtractor(
            extractor_model,
            max_keypoints=max_keypoints,
            min_score=min_score,
            providers=providers,
            num_threads=num_threads,
        )
        self.matcher = LightGlueMatcher(
            matcher_model,
            min_score=matcher_min_score,
            providers=providers,
            num_threads=num_threads,
        )

    @beartype
    def match_pair(
        self,
        image0: UInt8[np.ndarray, "h0 w0 3"],
        image1: UInt8[np.ndarray, "h1 w1 3"],
        *,
        max_side: int | None = None,
    ) -> dict[str, Shaped[np.ndarray, "m k"]]:
        del max_side  # the COLMAP backend always runs at native resolution
        features0 = self.extractor.extract(image0)
        features1 = self.extractor.extract(image1)
        if len(features0.points) == 0 or len(features1.points) == 0:
            empty_points = np.zeros((0, 2), dtype=np.float32)
            empty_descriptors = np.zeros((0, self.extractor.descriptor_dim), dtype=np.float32)
            return {
                "keypoints0": empty_points,
                "keypoints1": empty_points,
                "descriptors0": empty_descriptors,
                "descriptors1": empty_descriptors,
            }
        size0 = (image0.shape[1], image0.shape[0])
        size1 = (image1.shape[1], image1.shape[0])
        indices0, indices1, _ = self.matcher.match(
            features0.points,
            features1.points,
            features0.descriptors,
            features1.descriptors,
            size0,
            size1,
        )
        return {
            "keypoints0": features0.points[indices0],
            "keypoints1": features1.points[indices1],
            "descriptors0": features0.descriptors[indices0],
            "descriptors1": features1.descriptors[indices1],
        }


BACKENDS = {
    "raco": RaCoAlikedLightGlue,
    "aliked": AlikedLightGlue,
}
