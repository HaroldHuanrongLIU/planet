"""Train a minimal PlaNet-style RSSM world model on SurgWMBench."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from planet_torch_surgwmbench.data import (
    SurgWMBenchClipDataset,
    collate_dense_variable_length,
    collate_sparse_anchors,
)
from planet_torch_surgwmbench.models import PlaNetSurgWMBench
from planet_torch_surgwmbench.utils import load_config, recursive_update, seed_everything

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda value, **_: value


def default_config() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "SurgWMBench",
            "root": "/mnt/hdd1/neurips2026_dataset_track/SurgWMBench",
            "train_manifest": "manifests/train.jsonl",
            "val_manifest": "manifests/val.jsonl",
            "image_size": 128,
            "frame_sampling": "sparse_anchors",
            "interpolation_method": "linear",
            "use_dense_pseudo": False,
        },
        "encoder": {
            "embed_dim": 512,
            "channels": [32, 64, 128, 256],
            "activation": "elu",
        },
        "decoder": {
            "channels": [256, 128, 64, 32],
            "activation": "elu",
        },
        "rssm": {
            "stoch_dim": 32,
            "deter_dim": 200,
            "hidden_dim": 200,
            "action_dim": 3,
            "min_std": 0.1,
            "activation": "elu",
        },
        "coordinate_head": {
            "hidden_dim": 256,
            "output_dim": 2,
            "sigmoid_output": True,
        },
        "loss": {
            "recon_weight": 1.0,
            "kl_weight": 1.0,
            "free_nats": 3.0,
            "sparse_coord_weight": 10.0,
            "smoothness_weight": 0.1,
        },
        "train": {
            "batch_size": 4,
            "num_workers": 4,
            "epochs": 1,
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "precision": "amp",
            "seed": 42,
            "grad_clip_norm": 100.0,
            "compile": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-sampling", choices=["sparse_anchors", "dense", "all", "window"], default=None)
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--use-dense-pseudo", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=None)
    parser.add_argument("--debug-shapes", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = default_config()
    recursive_update(config, load_config(args.config))

    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    if args.dataset_root is not None:
        dataset_cfg["root"] = str(args.dataset_root)
    if args.train_manifest is not None:
        dataset_cfg["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        dataset_cfg["val_manifest"] = args.val_manifest
    if args.frame_sampling is not None:
        dataset_cfg["frame_sampling"] = args.frame_sampling
    if args.interpolation_method is not None:
        dataset_cfg["interpolation_method"] = args.interpolation_method
    if args.use_dense_pseudo:
        dataset_cfg["use_dense_pseudo"] = True
    if args.image_size is not None:
        dataset_cfg["image_size"] = int(args.image_size)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.lr is not None:
        train_cfg["lr"] = float(args.lr)
    return config


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    dataset_cfg = config["dataset"]
    manifest_key = "train_manifest" if split == "train" else "val_manifest"
    frame_sampling = dataset_cfg.get("frame_sampling", "sparse_anchors")
    dataset = SurgWMBenchClipDataset(
        dataset_root=dataset_cfg["root"],
        manifest=dataset_cfg[manifest_key],
        interpolation_method=dataset_cfg.get("interpolation_method", "linear"),
        image_size=int(dataset_cfg.get("image_size", 128)),
        frame_sampling=frame_sampling,
        use_dense_pseudo=bool(dataset_cfg.get("use_dense_pseudo", False)),
        strict=True,
    )
    collate_fn = collate_sparse_anchors if frame_sampling == "sparse_anchors" else collate_dense_variable_length
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def limit_batches(loader: Iterable[dict[str, Any]], max_batches: int | None) -> Iterable[dict[str, Any]]:
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(sum(item.get(key, 0.0) for item in items) / len(items)) for key in keys}


def train_one_epoch(
    model: PlaNetSurgWMBench,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None,
    debug_shapes: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    train_cfg = config["train"]
    use_amp = train_cfg.get("precision", "amp") == "amp" and device.type == "cuda"
    metrics: list[dict[str, float]] = []
    steps = 0
    iterator = tqdm(limit_batches(loader, max_batches), desc="train", leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        if debug_shapes and steps == 0:
            logging.info("frames=%s actions=%s coords=%s", tuple(batch["frames"].shape), tuple(batch["actions_delta_dt"].shape), tuple(batch["coords_norm"].shape))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, item_metrics = model.compute_losses(batch, config.get("loss", {}))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_clip = float(train_cfg.get("grad_clip_norm", 100.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        metrics.append(item_metrics)
        steps += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{item_metrics['loss']:.4f}", coord=f"{item_metrics['loss_sparse_human']:.4f}")
    return mean_metrics(metrics), steps


@torch.inference_mode()
def validate(
    model: PlaNetSurgWMBench,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for batch in tqdm(limit_batches(loader, max_batches), desc="val", leave=False):
        batch = move_batch(batch, device)
        _, item_metrics = model.compute_losses(batch, config.get("loss", {}))
        metrics.append(item_metrics)
    return mean_metrics(metrics)


def save_checkpoint(
    path: Path,
    model: PlaNetSurgWMBench,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "dataset_name": "SurgWMBench",
        "manifest_paths": {
            "train": config["dataset"].get("train_manifest"),
            "val": config["dataset"].get("val_manifest"),
        },
    }
    torch.save(payload, path)


def main() -> None:
    setup_logging()
    args = parse_args()
    config = build_config(args)
    seed_everything(int(config["train"].get("seed", 42)))
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)
    logging.info("Dataset root: %s", config["dataset"]["root"])

    train_loader = build_loader(config, "train", shuffle=True)
    val_loader = build_loader(config, "val", shuffle=False)
    overfit_batches = args.overfit_batches
    if overfit_batches is not None:
        cached = [batch for batch in limit_batches(train_loader, overfit_batches)]
        train_iterable: Iterable[dict[str, Any]] = cached
        logging.info("Overfitting on %d cached batch(es)", len(cached))
    else:
        train_iterable = train_loader

    model = PlaNetSurgWMBench(config).to(device)
    if bool(config["train"].get("compile", False)):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-5)),
    )
    use_amp = config["train"].get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    all_metrics: dict[str, Any] = {}
    for epoch in range(1, int(config["train"].get("epochs", 1)) + 1):
        train_metrics, steps = train_one_epoch(
            model=model,
            loader=train_iterable,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            max_batches=args.max_train_batches,
            debug_shapes=args.debug_shapes,
        )
        global_step += steps
        val_metrics = validate(model, val_loader, device, config, args.max_val_batches)
        all_metrics = {"train": train_metrics, "val": val_metrics}
        logging.info("epoch=%d train=%s", epoch, json.dumps(train_metrics, sort_keys=True))
        logging.info("epoch=%d val=%s", epoch, json.dumps(val_metrics, sort_keys=True))

    save_checkpoint(args.output, model, optimizer, config, int(config["train"].get("epochs", 1)), global_step, all_metrics)
    logging.info("Saved checkpoint: %s", args.output)


if __name__ == "__main__":
    main()
