"""Coordinate prediction head for SurgWMBench RSSM features."""

from __future__ import annotations

import torch
from torch import nn

from planet_torch_surgwmbench.models.encoder import _activation


class CoordinateHead(nn.Module):
    """Predict normalized instrument coordinates from RSSM features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 2,
        activation: str = "elu",
        sigmoid_output: bool = True,
    ) -> None:
        super().__init__()
        self.sigmoid_output = bool(sigmoid_output)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            _activation(activation),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            _activation(activation),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        coords = self.net(features)
        if self.sigmoid_output:
            coords = torch.sigmoid(coords)
        return coords
