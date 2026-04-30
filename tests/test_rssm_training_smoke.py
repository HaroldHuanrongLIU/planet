from __future__ import annotations

import torch

from planet_torch_surgwmbench.data import SurgWMBenchClipDataset, collate_sparse_anchors
from planet_torch_surgwmbench.models import PlaNetSurgWMBench
from planet_torch_surgwmbench.training.train_rssm import default_config
from tests.surgwmbench_test_utils import make_surgwmbench_root


def test_planet_rssm_loss_backward_on_toy_surgwmbench(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    config = default_config()
    config["dataset"].update(
        {
            "root": str(root),
            "train_manifest": "manifests/train.jsonl",
            "val_manifest": "manifests/val.jsonl",
            "image_size": 32,
        }
    )
    config["encoder"].update({"embed_dim": 64, "channels": [8, 16, 32, 64]})
    config["decoder"].update({"channels": [64, 32, 16, 8]})
    config["rssm"].update({"stoch_dim": 8, "deter_dim": 32, "hidden_dim": 32, "embed_dim": 64})
    config["coordinate_head"].update({"hidden_dim": 32})

    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )
    batch = collate_sparse_anchors([dataset[0]])
    model = PlaNetSurgWMBench(config)
    loss, metrics = model.compute_losses(batch, config["loss"])
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss_recon"] > 0
    assert metrics["loss_kl"] >= 0
    assert metrics["loss_sparse_human"] >= 0
