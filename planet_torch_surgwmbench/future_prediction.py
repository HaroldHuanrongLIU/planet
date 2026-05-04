from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
from torch import nn

from planet_torch_surgwmbench.models.planet_surgwmbench import PlaNetSurgWMBench
from planet_torch_surgwmbench.models.rssm import RSSMState
from surgwmbench_benchmark.future_model_helpers import context_delta_actions, zero_actions
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class PlaNetFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the PlaNet RSSM core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = PlaNetSurgWMBench(
            {
                "dataset": {"image_size": config.image_size},
                "encoder": {"embed_dim": max(config.hidden_dim, 128)},
                "rssm": {
                    "stoch_dim": max(config.latent_dim // 2, 16),
                    "deter_dim": config.hidden_dim,
                    "hidden_dim": config.hidden_dim,
                    "action_dim": 3,
                },
                "coordinate_head": {"hidden_dim": config.hidden_dim},
            }
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context_actions = context_delta_actions(batch["context_coords_norm"], batch["context_frame_indices"], action_dim=3)
        posterior, _ = self.core.observe(self.core.encode(batch["context_frames"]), context_actions, deterministic=False)
        state = RSSMState(
            mean=posterior.mean[:, -1],
            std=posterior.std[:, -1],
            stoch=posterior.stoch[:, -1],
            deter=posterior.deter[:, -1],
        )
        imagined = self.core.rssm.imagine(state, zero_actions(batch, action_dim=3), deterministic=False)
        pred_frames = self.core.decode_obs(imagined)
        pred_coords = self.core.decode_coord(imagined)
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return PlaNetFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("planet", "PlaNetFuturePredictionCore", "planet_torch_surgwmbench.data.surgwmbench", make_model))
