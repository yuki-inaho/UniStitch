#!/usr/bin/env python3
"""Export the v3.0 ALIKED descriptor head to ONNX.

The released ``raco_aliked_lightglue_pipeline_k*.onnx`` models from
`fabio-sim/LightGlue-ONNX v3.0 <https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v3.0>`_
expose ``(keypoints, matches, mscores)`` only. UniStitch's point backbone also
consumes per-keypoint descriptors, so this script exports the descriptor half of
the same pipeline: ALIKED's dense features sampled at arbitrary keypoints
(``(images, keypoints) -> descriptors``, 128-d).

Using the *released* pipeline for detection/matching and this model only for
descriptors keeps the keypoints bit-identical to the pipeline output, so match
indices map 1:1 onto the descriptors.

Build-time dependencies (not needed at runtime):
  * torch, onnx, onnxscript, onnxruntime
  * lightglue_dynamo from LightGlue-ONNX (``--lightglue_dynamo_path``, e.g.
    ``git clone --depth 1 https://github.com/fabio-sim/LightGlue-ONNX``)
  * ALIKED weights (https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth,
    BSD-3-Clause)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "models" / "aliked-descriptor.onnx"

INPUT_DIM_DIVISOR = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DescriptorWrapper(torch.nn.Module):
    """Sample ALIKED descriptors at given keypoints."""

    def __init__(self, descriptor: torch.nn.Module) -> None:
        super().__init__()
        self.descriptor = descriptor

    def forward(
        self,
        images: Float[torch.Tensor, "two_b 3 h w"],
        keypoints: Float[torch.Tensor, "two_b k 2"],
    ) -> Float[torch.Tensor, "two_b k 128"]:
        return self.descriptor(images, keypoints)


@beartype
def build_dynamic_shapes() -> tuple[dict[int, Any], ...]:
    """Dynamic spatial dimensions and keypoint count; batch stays at one pair."""
    height_factor = torch.export.Dim("height_factor", min=2)
    width_factor = torch.export.Dim("width_factor", min=2)
    num_keypoints = torch.export.Dim("num_keypoints", min=1)
    return (
        {
            2: INPUT_DIM_DIVISOR * height_factor,
            3: INPUT_DIM_DIVISOR * width_factor,
        },
        {1: num_keypoints},
    )


def export(args: argparse.Namespace) -> Path:
    import onnx
    from onnxscript import opset20 as onnx_op

    if str(args.lightglue_dynamo_path) not in sys.path:
        sys.path.insert(0, str(args.lightglue_dynamo_path))

    from lightglue_dynamo.models.aliked import ALIKEDDescriptor  # ty: ignore[unresolved-import]

    descriptor = ALIKEDDescriptor(portable_deform_conv=True)
    descriptor.eval()
    if args.fuse_batch_norm:
        descriptor.fuse_batch_norm()
    model = DescriptorWrapper(descriptor).eval()

    def translate_integer_div(self: object, other: object, rounding_mode: str | None = None) -> object:
        if rounding_mode not in {"floor", "trunc"}:
            raise ValueError(f"Unsupported integer division mode: {rounding_mode}")
        # ONNX integer Div is exact for non-negative operands; the default
        # decomposition casts through float and can overflow large indices.
        return onnx_op.Div(self, other)  # ty: ignore[invalid-argument-type]

    example_images = torch.zeros(2, 3, 1024, 1024)
    example_keypoints = torch.zeros(2, 2048, 2)

    torch.onnx.export(
        model,
        (example_images, example_keypoints),
        str(args.output),
        input_names=["images", "keypoints"],
        output_names=["descriptors"],
        opset_version=20,
        dynamic_shapes=build_dynamic_shapes(),
        dynamo=True,
        external_data=False,
        optimize=False,
        custom_translation_table={torch.ops.aten.div.Tensor_mode: translate_integer_div},
    )
    onnx.checker.check_model(str(args.output))
    print(f"exported: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256:   {sha256_file(args.output)}")
    return args.output


@beartype
def validate(
    output: Path,
    lightglue_dynamo_path: Path,
    image_paths: tuple[Path, Path],
    size: int = 1024,
) -> None:
    """Compare ONNX descriptors against the torch reference at random keypoints."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    if str(lightglue_dynamo_path) not in sys.path:
        sys.path.insert(0, str(lightglue_dynamo_path))
    from lightglue_dynamo.models.aliked import ALIKEDDescriptor  # ty: ignore[unresolved-import]

    images = [
        cv2.resize(cv2.imread(str(path), cv2.IMREAD_COLOR), (size, size), interpolation=cv2.INTER_AREA)  # ty: ignore[no-matching-overload]
        for path in image_paths
    ]
    blob = cv2.dnn.blobFromImages(
        images, scalefactor=1 / 255.0, size=(size, size), swapRB=True, crop=False, ddepth=cv2.CV_32F
    ).astype(np.float32)

    rng = np.random.default_rng(0)
    keypoints = rng.uniform(4, size - 4, size=(2, 512, 2)).astype(np.float32)

    reference_descriptor = ALIKEDDescriptor(portable_deform_conv=True).eval()
    reference_descriptor.fuse_batch_norm()
    with torch.no_grad():
        reference = reference_descriptor(torch.from_numpy(blob), torch.from_numpy(keypoints)).numpy()

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(output), session_options, providers=["CPUExecutionProvider"])
    (descriptors,) = session.run(None, {"images": blob, "keypoints": keypoints})
    descriptors = np.asarray(descriptors, dtype=np.float32)

    difference = np.abs(descriptors - reference).max()
    print(f"validation: descriptors shape={descriptors.shape} max |onnx - torch| = {difference:.6g}")
    if difference > 1e-4:
        raise SystemExit(f"ONNX descriptors differ from the torch reference (max diff {difference})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lightglue_dynamo_path", type=Path, required=True, help="path to a LightGlue-ONNX checkout")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate_image1", type=Path, default=None)
    parser.add_argument("--validate_image2", type=Path, default=None)
    parser.add_argument("--no_fuse_batch_norm", dest="fuse_batch_norm", action="store_false")
    parser.set_defaults(fuse_batch_norm=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(args)

    if args.validate_image1 is not None and args.validate_image2 is not None:
        validate(args.output, args.lightglue_dynamo_path, (args.validate_image1, args.validate_image2))


if __name__ == "__main__":
    main()
