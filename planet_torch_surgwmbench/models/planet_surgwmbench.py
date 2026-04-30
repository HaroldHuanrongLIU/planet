"""Minimal trainable PlaNet-style world model for SurgWMBench."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from planet_torch_surgwmbench.models.coord_head import CoordinateHead
from planet_torch_surgwmbench.models.decoder import ObservationDecoder
from planet_torch_surgwmbench.models.encoder import ObservationEncoder
from planet_torch_surgwmbench.models.rssm import RSSM, RSSMState, kl_divergence


class PlaNetSurgWMBench(nn.Module):
    """World-model baseline with image reconstruction and coordinate decoding."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        encoder_cfg = config.get("encoder", {})
        rssm_cfg = config.get("rssm", {})
        decoder_cfg = config.get("decoder", {})
        coord_cfg = config.get("coordinate_head", {})
        dataset_cfg = config.get("dataset", {})

        image_size = int(dataset_cfg.get("image_size", 128))
        embed_dim = int(encoder_cfg.get("embed_dim", 1024))
        self.encoder = ObservationEncoder(
            image_size=image_size,
            embed_dim=embed_dim,
            channels=encoder_cfg.get("channels", (32, 64, 128, 256)),
            activation=encoder_cfg.get("activation", "elu"),
        )
        self.rssm = RSSM(
            stoch_dim=int(rssm_cfg.get("stoch_dim", 32)),
            deter_dim=int(rssm_cfg.get("deter_dim", 200)),
            hidden_dim=int(rssm_cfg.get("hidden_dim", 200)),
            action_dim=int(rssm_cfg.get("action_dim", 3)),
            embed_dim=embed_dim,
            min_std=float(rssm_cfg.get("min_std", 0.1)),
            activation=rssm_cfg.get("activation", "elu"),
        )
        self.decoder = ObservationDecoder(
            feature_dim=self.rssm.feature_dim,
            image_size=image_size,
            channels=decoder_cfg.get("channels", (256, 128, 64, 32)),
            activation=decoder_cfg.get("activation", encoder_cfg.get("activation", "elu")),
        )
        self.coord_head = CoordinateHead(
            feature_dim=self.rssm.feature_dim,
            hidden_dim=int(coord_cfg.get("hidden_dim", 256)),
            output_dim=int(coord_cfg.get("output_dim", 2)),
            activation=coord_cfg.get("activation", encoder_cfg.get("activation", "elu")),
            sigmoid_output=bool(coord_cfg.get("sigmoid_output", True)),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, deterministic: bool = False) -> tuple[RSSMState, RSSMState]:
        return self.rssm.observe(embeds, actions, deterministic=deterministic)

    def decode_obs(self, states: RSSMState) -> torch.Tensor:
        return self.decoder(self.rssm.features(states))

    def decode_coord(self, states: RSSMState) -> torch.Tensor:
        return self.coord_head(self.rssm.features(states))

    def compute_losses(self, batch: dict[str, Any], loss_cfg: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        loss_cfg = loss_cfg or {}
        frames = batch["frames"]
        coords = batch["coords_norm"]
        actions = batch["actions_delta_dt"]
        frame_mask = batch.get("frame_mask", batch.get("human_anchor_mask", batch.get("mask")))
        label_weight = batch.get("label_weight")

        embeds = self.encode(frames)
        post, prior = self.observe(embeds, actions, deterministic=False)
        features = self.rssm.features(post)
        recon = self.decoder(features)
        pred_coords = self.coord_head(features)

        if frame_mask is None:
            frame_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
        else:
            frame_mask = frame_mask.to(device=frames.device, dtype=torch.bool)
        if label_weight is None:
            label_weight = frame_mask.to(dtype=frames.dtype)
        else:
            label_weight = label_weight.to(device=frames.device, dtype=frames.dtype) * frame_mask.to(dtype=frames.dtype)

        recon_per_frame = F.mse_loss(recon, frames, reduction="none").mean(dim=(-3, -2, -1))
        recon_loss = (recon_per_frame * frame_mask.to(dtype=frames.dtype)).sum() / frame_mask.sum().clamp_min(1)

        coord_per_frame = F.smooth_l1_loss(pred_coords, coords, reduction="none").sum(dim=-1)
        coord_loss = (coord_per_frame * label_weight).sum() / label_weight.sum().clamp_min(1.0)

        kl = kl_divergence(post, prior)
        free_nats = float(loss_cfg.get("free_nats", 0.0))
        if free_nats > 0:
            kl = torch.clamp(kl, min=free_nats)
        kl_loss = (kl * frame_mask.to(dtype=kl.dtype)).sum() / frame_mask.sum().clamp_min(1)

        smoothness_loss = torch.zeros((), device=frames.device, dtype=frames.dtype)
        if pred_coords.shape[1] >= 3:
            second_diff = pred_coords[:, 2:] - 2.0 * pred_coords[:, 1:-1] + pred_coords[:, :-2]
            smooth_mask = frame_mask[:, 2:] & frame_mask[:, 1:-1] & frame_mask[:, :-2]
            if bool(smooth_mask.any()):
                smoothness_loss = (second_diff.square().sum(dim=-1) * smooth_mask.to(dtype=frames.dtype)).sum()
                smoothness_loss = smoothness_loss / smooth_mask.sum().clamp_min(1)

        recon_weight = float(loss_cfg.get("recon_weight", 1.0))
        kl_weight = float(loss_cfg.get("kl_weight", 1.0))
        sparse_coord_weight = float(loss_cfg.get("sparse_coord_weight", 10.0))
        smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
        loss = (
            recon_weight * recon_loss
            + kl_weight * kl_loss
            + sparse_coord_weight * coord_loss
            + smoothness_weight * smoothness_loss
        )

        metrics = {
            "loss": float(loss.detach().item()),
            "loss_recon": float(recon_loss.detach().item()),
            "loss_kl": float(kl_loss.detach().item()),
            "loss_sparse_human": float(coord_loss.detach().item()),
            "loss_dense_pseudo": 0.0,
            "loss_smoothness": float(smoothness_loss.detach().item()),
            "coord_mae": float((torch.abs(pred_coords.detach() - coords) * label_weight[..., None]).sum().item() / label_weight.sum().clamp_min(1.0).item() / 2.0),
        }
        return loss, metrics
