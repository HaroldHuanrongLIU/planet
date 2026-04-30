"""Image observation decoder for PlaNet-style reconstruction losses."""

from __future__ import annotations

import torch
from torch import nn

from planet_torch_surgwmbench.models.encoder import _activation


class ObservationDecoder(nn.Module):
    """Decode RSSM features into image reconstructions."""

    def __init__(
        self,
        feature_dim: int,
        image_size: int = 128,
        channels: list[int] | tuple[int, ...] = (256, 128, 64, 32),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.image_size = int(image_size)
        self.channels = tuple(int(value) for value in channels)
        if self.image_size % (2 ** len(self.channels)) != 0:
            raise ValueError("image_size must be divisible by 2 ** len(decoder.channels)")
        self.start_size = self.image_size // (2 ** len(self.channels))
        self.fc = nn.Linear(self.feature_dim, self.channels[0] * self.start_size * self.start_size)

        layers: list[nn.Module] = []
        in_channels = self.channels[0]
        for out_channels in self.channels[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(_activation(activation))
            in_channels = out_channels
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,D]`` or ``[B,T,D]`` features."""

        is_sequence = features.ndim == 3
        if is_sequence:
            batch, time = features.shape[:2]
            x = features.reshape(batch * time, features.shape[-1])
        elif features.ndim == 2:
            batch, time = features.shape[0], None
            x = features
        else:
            raise ValueError(f"features must have shape [B,D] or [B,T,D], got {tuple(features.shape)}")

        hidden = self.fc(x).reshape(x.shape[0], self.channels[0], self.start_size, self.start_size)
        recon = self.deconv(hidden)
        if is_sequence:
            recon = recon.reshape(batch, time, *recon.shape[1:])
        return recon
