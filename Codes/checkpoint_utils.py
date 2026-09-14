"""Checkpoint loading helpers shared by training, inference and notebooks.

The released UniStitch checkpoint was trained with 256-d SuperPoint
descriptors. To fine-tune with a smaller descriptor (e.g. ALIKED, 128-d) two
options exist:

* keep the 256-d stem and zero-pad the descriptors
  (``data.descriptor_pad_dim=256``), or
* build a native ``descriptor_dim=128`` network and initialise the stem from the
  pretrained weights: keep the xy columns and the first descriptor columns.

``load_model_state`` implements the second option transparently (and loads every
other tensor by exact shape match).
"""

from __future__ import annotations

from typing import Any

import torch

STEM_WEIGHT_SUFFIX = "pointnext_feat.encoder.stem.0.weight"


def infer_descriptor_dim(state: dict[str, Any]) -> int:
    """Read the descriptor dimension from a checkpoint's stem weight."""
    for key, value in state.items():
        if key.endswith(STEM_WEIGHT_SUFFIX):
            return int(value.shape[1]) - 2
    raise KeyError(f"no stem weight ({STEM_WEIGHT_SUFFIX}) found in the checkpoint")


def _truncate_stem_weight(target: torch.Tensor, source: torch.Tensor) -> str | None:
    """Copy xy + the first descriptor columns of a wider pretrained stem."""
    if source.ndim != target.ndim or source.shape[0] != target.shape[0]:
        return f"stem shape mismatch {tuple(source.shape)} -> {tuple(target.shape)}"
    if source.shape[1] < target.shape[1]:
        return f"pretrained stem is narrower ({source.shape[1]}) than the model ({target.shape[1]})"
    target[:, :2].copy_(source[:, :2])
    target[:, 2:].copy_(source[:, 2 : 2 + target.shape[1] - 2])
    return None


def load_model_state(net: torch.nn.Module, state: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Load ``state`` into ``net``, adapting the descriptor stem if needed.

    Tensors whose shape matches are copied; a wider pretrained stem is truncated
    to the model's descriptor dimension (xy columns + first descriptor columns).
    Returns a report with the loaded/skipped/missing keys.
    """
    model_state = net.state_dict()
    loaded = 0
    skipped: list[str] = []
    stem_resizes: list[str] = []

    for key, value in state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            target.copy_(value)
            loaded += 1
        elif key.endswith(STEM_WEIGHT_SUFFIX):
            error = _truncate_stem_weight(target, value)
            if error is None:
                loaded += 1
                stem_resizes.append(f"{value.shape[1]} -> {target.shape[1]}")
            else:
                skipped.append(f"{key}: {error}")
        else:
            skipped.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")

    missing = [key for key in model_state if key not in state]
    if stem_resizes:
        print(f"stem warm start (descriptor channels): {', '.join(stem_resizes)}")
    for entry in skipped[:8]:
        print(f"checkpoint tensor skipped (shape mismatch): {entry}")
    print(f"loaded {loaded} tensors, skipped {len(skipped)}, missing {len(missing)} (strict={strict})")
    if strict and (skipped or missing):
        raise RuntimeError(f"strict load failed: skipped={len(skipped)} missing={len(missing)}")
    return {"loaded": loaded, "skipped": skipped, "missing": missing, "stem_resizes": stem_resizes}
