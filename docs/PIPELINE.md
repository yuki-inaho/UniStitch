# pixi / ONNX pipeline guide

This fork adds a reproducible Pixi environment, a self-contained ONNX keypoint
pipeline (RaCo-ALIKED-LightGlue+ v3.0 and COLMAP-style ALIKED+LightGlue), and
Hydra-managed fine-tuning with the [AMUSE](https://github.com/kjeiun/amuse)
optimizer.

## 1. Environment

```bash
pixi install                       # creates .pixi/envs/default
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Key tasks:

| task | command |
| --- | --- |
| `pixi run train` | Hydra training entry point (`Codes/train.py`) |
| `pixi run infer` | Hydra inference/qualitative evaluation (`Codes/infer.py`) |
| `pixi run tensorboard` | TensorBoard over `outputs/` (http://localhost:6006) |
| `pixi run keypoints` | ONNX keypoint tool (`keypoint_tool/onnx_keypoint_tool.py`) |
| `pixi run test` | fast pytest smoke tests for the ONNX pipeline |
| `pixi run lint` / `format` / `typecheck` | Ruff, Ruff format, `ty` |

## 2. Pretrained checkpoint

```bash
mkdir -p model_homo_stage2
curl -L -o model_homo_stage2/epoch_best_model.pth \
  https://huggingface.co/Y5Y/UniStitch_model/resolve/main/epoch_best_model.pth
```

The released checkpoint is a stage-2 model trained with 256-d SuperPoint
descriptors (`point_backbone.pointnext_feat.encoder.stem.0.weight` has shape
`(64, 258, 1)`).

## 3. Self-contained ONNX keypoints

Model files are vendored under `keypoint_tool/models/` (checksums in
`keypoint_tool/models/SHA256SUMS`), so no runtime downloads are needed:

| model | source | license |
| --- | --- | --- |
| `raco_aliked_lightglue_pipeline_k2048.onnx` | [fabio-sim/LightGlue-ONNX v3.0](https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v3.0) | Apache-2.0 (LightGlue); RaCo detector from [cvg/RaCo](https://github.com/cvg/RaCo) |
| `aliked-descriptor.onnx` | exported with `keypoint_tool/export_aliked_descriptor.py` (LightGlue-ONNX v3.0 code) | ALIKED weights BSD-3-Clause ([Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED)) |
| `aliked-n16rot.onnx` / `aliked-lightglue.onnx` | [COLMAP 3.13.0 release assets](https://github.com/colmap/colmap/releases/tag/3.13.0) | BSD-3-Clause / Apache-2.0 |

The released v3.0 pipeline outputs `(keypoints, matches, mscores)` only. The
companion `aliked-descriptor.onnx` samples the same ALIKED 128-d descriptors at
those keypoints, which is what UniStitch's point backbone consumes.

Build a keypoint database (lmdb) for a dataset folder with `input1/` + `input2/`:

```bash
# default backend: RaCo-ALIKED-LightGlue+ v3.0
pixi run keypoints build --data_path data/UDIS-D/training --workers 16

# COLMAP-style ALIKED extractor + ALIKED-LightGlue matcher
pixi run keypoints build --backend aliked --data_path data/UDIS-D/training
```

The database is written to `<data_path>/aliked_lmdb` with the exact key layout
expected by `Codes/dataset.py` (`{index:08d}_{image_name}`). Runs are resumable
(entries are committed every `--commit_every` pairs).

### GPU extraction (optional, much faster)

The default install uses CPU ONNX Runtime. For GPU extraction install the CUDA
wheel and expose the CUDA/cuDNN libraries that ship with the PyTorch wheel:

```bash
pixi run pip install onnxruntime-gpu
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib/python3.11/site-packages/nvidia/cu13/lib:\
$PWD/.pixi/envs/default/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=1 pixi run keypoints build --data_path data/UDIS-D/training --workers 2 --ort_threads 1
```

Two workers saturate a 16 GB GPU (~10 pairs/s at `--max_side 1024`); more
workers exhaust GPU memory.

To re-export the descriptor model from scratch:

```bash
git clone --depth 1 https://github.com/fabio-sim/LightGlue-ONNX
pixi run python keypoint_tool/export_aliked_descriptor.py \
  --lightglue_dynamo_path LightGlue-ONNX
```

## 4. Data layout

```
data/
├── UDIS-D/
│   ├── training/{input1,input2,aliked_lmdb}
│   └── testing/{input1,input2,aliked_lmdb}
└── classical_tmp/stitch_real/{input1,input2,aliked_lmdb}
```

`data/` is git-ignored. The UDIS-D zips can be downloaded from the
[UDIS project page](https://github.com/nie-lang/UnsupervisedDeepImageStitching);
the classical pairs come from [RopStitch](https://github.com/MmelodYy/RopStitch).

## 5. Fine-tuning with AMUSE (Hydra)

The released checkpoint expects 256-d descriptors; ALIKED emits 128-d. The
dataset zero-pads descriptors to `data.descriptor_pad_dim=256`
(`Codes/dataset.py:pad_descriptor_dim`), so every pretrained weight loads and
the point backbone learns to use the first 128 channels.

```bash
# fine-tune stage 2 from the released checkpoint on GPU 1 (default)
pixi run train checkpoint.pretrained=model_homo_stage2/epoch_best_model.pth

# classic AdamW baseline
pixi run train optim=adamw optim.lr=1e-4 epochs=20

# resume (AMUSE schedule-free state is restored as well)
pixi run train checkpoint.resume=outputs/2026-09-14/17-30-00/checkpoints/last.pth
```

Defaults live in `Codes/configs/` (`train.yaml`, `data/udis.yaml`,
`optim/amuse.yaml`, `optim/adamw.yaml`, `stage/stage2.yaml`); everything can be
overridden on the command line (`data.batch_size=8 optim.lr=0.01 epochs=5 ...`).

- **Checkpoints**: only the `save.keep_best_k` best checkpoints by validation
  SSIM are kept (`checkpoints/best_epoch*_ssim*.pth`, with `best.pth` symlinked
  to the current best), plus `checkpoints/last.pth` for resuming.
- **AMUSE**: parameters are split into a Muon group (hidden matrices) and an
  AdamW-style group (biases/norms/regression heads,
  `optim.head_param_patterns`). The optimizer's `train()`/`eval()` iterate
  conversion is handled around validation/checkpointing, so saved weights are
  always the evaluation (averaged) iterate.

### TensorBoard

```bash
pixi run tensorboard                 # http://localhost:6006, logs under outputs/
pixi run tensorboard --logdir outputs/2026-09-14
```

Logged: total/overlap/non-overlap/df losses, per-group learning rates, gradient
norm, AMUSE `beta1`, and validation SSIM (UDIS-D and classical) with sample
stitches.

## 6. Inference

```bash
# 10 UDIS-D testing pairs with the released (SuperPoint) checkpoint
pixi run infer checkpoint=model_homo_stage2/epoch_best_model.pth limit=10

# all pairs with a fine-tuned ALIKED checkpoint
pixi run infer data.keypoint=aliked data.descriptor_pad_dim=256 \
  checkpoint=outputs/.../checkpoints/best.pth limit=null
```

Fused images, per-pair SSIM/PSNR logs, and `metrics.json` are written to the
Hydra run directory (`outputs/infer/...`).

Large images (e.g. the classical pairs) can exceed GPU memory inside the FFD/TPS
warping step, so `max_side: 1600` resizes inputs (and rescales keypoints) before
inference. Set `max_side=null` to evaluate at native resolution (needs more
VRAM).

## 7. Notebooks

`notebooks/matching_and_stitching_epoch0.ipynb` demonstrates matching
(RaCo-ALIKED-LightGlue+, ONNX) and stitching (fine-tuned vs. released baseline)
on sample UDIS-D/classical pairs using the end-of-epoch-0 checkpoint. Outputs
are embedded; figures are also written to `notebooks/figures/`.

```bash
pixi run lab   # interactive
# or headless:
pixi run jupyter nbconvert --to notebook --execute --inplace notebooks/matching_and_stitching_epoch0.ipynb
```

The ONNX matcher is forced to CPU inside the notebook because the training run
occupies GPU1; the stitching network uses GPU1.
