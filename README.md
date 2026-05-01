# PlaNet for SurgWMBench

This repository starts from the official Google Research PlaNet implementation
of **Learning Latent Dynamics for Planning from Pixels** and adds a modern
PyTorch 2.x baseline path for:

**SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning**

The original PlaNet code under `planet/` is TensorFlow 1.x code and is kept for
reference. The SurgWMBench adaptation lives in `planet_torch_surgwmbench/` and
does not patch or depend on TensorFlow.

## Current Status

Implemented:

- SurgWMBench manifest-based clip loader.
- Sparse 20-human-anchor mode.
- Dense variable-length interpolation mode.
- Frame and raw-video window datasets.
- Collate functions that produce action deltas `[dx_norm, dy_norm, dt]`.
- Trajectory metrics for sparse/dense coordinate evaluation.
- Toy SurgWMBench generator and read-only dataset validator.
- A trainable PyTorch PlaNet-style RSSM smoke baseline:
  - image encoder
  - action-conditioned RSSM
  - stochastic and deterministic latent state
  - observation decoder
  - coordinate decoder
  - reconstruction, KL, coordinate, and smoothness losses

Still pending:

- CEM latent planner.
- Policy controller.
- Sparse rollout evaluation script.
- Dense auxiliary evaluation script.
- Visualization scripts.
- Full training/evaluation documentation for all stages.

## Repository Layout

```text
planet/
  Original TensorFlow 1.x PlaNet implementation.

planet_torch_surgwmbench/
  data/          SurgWMBench loaders, transforms, collators.
  models/        PyTorch encoder, decoder, RSSM, coordinate head.
  training/      RSSM training entry point.
  evaluation/    Trajectory metrics.
  utils/         Config and seeding helpers.

configs/
  surgwmbench_planet_rssm.yaml

tools/
  make_toy_surgwmbench.py
  validate_surgwmbench_loader.py

tests/
  Loader, collate, metrics, and RSSM smoke tests.
```

## Installation

Use Python 3.10+ and PyTorch 2.x. Install dependencies with:

```sh
python -m pip install -r requirements.txt
```

For CUDA builds of PyTorch, install the wheel matching your local CUDA/runtime
from the official PyTorch instructions if the generic `torch` package does not
match your machine.

## SurgWMBench Dataset

The local dataset root used on this machine is:

```sh
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Expected public layout:

```text
SurgWMBench/
  videos/
  clips/
  interpolations/
  manifests/
    train.jsonl
    val.jsonl
    test.jsonl
    all.jsonl
  metadata/
```

Rules used by this baseline:

- Use the official manifests. Do not create random splits.
- Every clip has exactly 20 human anchors.
- Clips are variable-length; do not assume 20 frames.
- Sparse human-anchor metrics are primary.
- Dense interpolation coordinates are auxiliary pseudo labels.
- Coordinates are `[x, y]` pixels with top-left origin.
- Training uses normalized coordinates internally.
- Pixel coordinates are preserved for reporting.

## Validate the Dataset

Run a read-only validation sample:

```sh
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 8
```

Expected output:

```text
SurgWMBench validation passed.
```

## Loader Smoke Test

```sh
python -c "from planet_torch_surgwmbench.data import SurgWMBenchClipDataset, collate_sparse_anchors; root='/mnt/hdd1/neurips2026_dataset_track/SurgWMBench'; ds=SurgWMBenchClipDataset(root,'manifests/train.jsonl',image_size=128,frame_sampling='sparse_anchors'); batch=collate_sparse_anchors([ds[0], ds[1]]); print(batch['frames'].shape, batch['actions_delta_dt'].shape)"
```

Expected shape pattern:

```text
torch.Size([2, 20, 3, 128, 128]) torch.Size([2, 19, 3])
```

## Train RSSM Smoke Baseline

The current training entry point is:

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

This command has been run successfully on the current machine with CUDA and a
real SurgWMBench subset. For full training, remove the `--max-train-batches`
and `--max-val-batches` limits and choose an output path outside `/tmp`.

## Tests

Run the focused PyTorch SurgWMBench tests:

```sh
pytest tests/test_surgwmbench_dataset.py \
  tests/test_surgwmbench_collate.py \
  tests/test_metrics.py \
  tests/test_rssm_training_smoke.py
```

## Original PlaNet Reference

The original PlaNet agent was introduced in:

```bibtex
@inproceedings{hafner2019planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle={International Conference on Machine Learning},
  pages={2555--2565},
  year={2019}
}
```

Original project links:

- [Google AI Blog post][blog]
- [Project website][website]
- [PDF paper][paper]

[blog]: https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html
[website]: https://danijar.com/project/planet/
[paper]: https://arxiv.org/pdf/1811.04551.pdf

The original TensorFlow dependencies remain documented in `setup.py`, but they
are not required for the PyTorch SurgWMBench adaptation.

Disclaimer: This is not an official Google product.
