"""CNN observation encoder for PlaNet-style SurgWMBench models."""

from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU(inplace=True)
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ObservationEncoder(nn.Module):
    """Encode images or image sequences into compact embeddings."""

    def __init__(
        self,
        image_size: int = 128,
        embed_dim: int = 1024,
        channels: list[int] | tuple[int, ...] = (32, 64, 128, 256),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, int(out_channels), kernel_size=4, stride=2, padding=1))
            layers.append(_activation(activation))
            in_channels = int(out_channels)
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(int(channels[-1]) * 4 * 4, self.embed_dim)
        self.out_act = _activation(activation)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,3,H,W]`` or ``[B,T,3,H,W]`` frames."""

        is_sequence = frames.ndim == 5
        if is_sequence:
            batch, time = frames.shape[:2]
            x = frames.reshape(batch * time, *frames.shape[2:])
        elif frames.ndim == 4:
            batch, time = frames.shape[0], None
            x = frames
        else:
            raise ValueError(f"frames must have shape [B,3,H,W] or [B,T,3,H,W], got {tuple(frames.shape)}")

        hidden = self.conv(x)
        hidden = self.pool(hidden).flatten(1)
        embed = self.out_act(self.proj(hidden))
        if is_sequence:
            embed = embed.reshape(batch, time, self.embed_dim)
        return embed
