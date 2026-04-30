# Repository Instructions

## Scope

This repository is a local clone of the original Google Research PlaNet codebase with an added PyTorch SurgWMBench adaptation.

- Keep the original `planet/` TensorFlow 1.x implementation readable and avoid patching it into a new baseline.
- Put new SurgWMBench/PyTorch work under `planet_torch_surgwmbench/`, `tools/`, `tests/`, `configs/`, or new documentation files.
- Do not use TensorFlow in the new PyTorch adaptation.

## SurgWMBench Data Contract

The canonical local dataset root is:

```sh
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Use official manifests only:

- `manifests/train.jsonl`
- `manifests/val.jsonl`
- `manifests/test.jsonl`
- `manifests/all.jsonl`

Do not create random train/val/test splits. Each clip has exactly 20 sparse human anchors, but variable-length frame sequences. Dense interpolation files provide auxiliary pseudo coordinates only; do not report them as human ground truth.

Coordinate conventions:

- Pixel coordinates are `[x, y]`, origin top-left.
- Use normalized coordinates internally for training.
- Preserve pixel coordinates for metrics/reporting.
- Do not silently clip coordinates.

## Current PyTorch First Pass

Implemented components:

- `planet_torch_surgwmbench/data/surgwmbench.py`
- `planet_torch_surgwmbench/data/raw_video.py`
- `planet_torch_surgwmbench/data/collate.py`
- `planet_torch_surgwmbench/models/`
- `planet_torch_surgwmbench/training/train_rssm.py`
- `planet_torch_surgwmbench/evaluation/metrics.py`
- `configs/surgwmbench_planet_rssm.yaml`
- `tools/make_toy_surgwmbench.py`
- `tools/validate_surgwmbench_loader.py`

The current training path supports a minimal PlaNet-style RSSM world-model smoke run. CEM, policy planning, dense auxiliary evaluation scripts, and full rollout evaluation are still pending.

## Validation Commands

Run focused tests:

```sh
pytest tests/test_surgwmbench_dataset.py tests/test_surgwmbench_collate.py tests/test_metrics.py
```

Validate the real dataset sample:

```sh
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 8
```

Loader/collate smoke:

```sh
python -c "from planet_torch_surgwmbench.data import SurgWMBenchClipDataset, collate_sparse_anchors; root='/mnt/hdd1/neurips2026_dataset_track/SurgWMBench'; ds=SurgWMBenchClipDataset(root,'manifests/train.jsonl',image_size=128,frame_sampling='sparse_anchors'); batch=collate_sparse_anchors([ds[0], ds[1]]); print(batch['frames'].shape, batch['actions_delta_dt'].shape)"
```

RSSM training smoke:

```sh
python -m planet_torch_surgwmbench.training.train_rssm \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --config configs/surgwmbench_planet_rssm.yaml \
  --output /tmp/planet_surgwmbench_rssm_smoke.pt \
  --batch-size 2 \
  --num-workers 2 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --debug-shapes
```

## Coding Style

- Use Python 3.10+ and PyTorch 2.x APIs.
- Use `pathlib`, type hints, dataclasses/configs where useful, and PIL for image loading.
- Keep optional OpenCV usage isolated to raw video loading.
- Use `torch.inference_mode()` for evaluation code and `torch.amp` for future training code.
- Keep `torch.compile` behind a config flag and disabled by default.

## Git Hygiene

- Stage only files relevant to the requested task.
- Do not commit generated caches such as `__pycache__/` or `.pytest_cache/`.
- Prefer pushing to `origin`, not `upstream`.
