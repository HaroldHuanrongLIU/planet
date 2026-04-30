"""Action-conditioned recurrent state-space model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from planet_torch_surgwmbench.models.encoder import _activation


@dataclass
class RSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor


def _stack_states(states: list[RSSMState]) -> RSSMState:
    return RSSMState(
        mean=torch.stack([state.mean for state in states], dim=1),
        std=torch.stack([state.std for state in states], dim=1),
        stoch=torch.stack([state.stoch for state in states], dim=1),
        deter=torch.stack([state.deter for state in states], dim=1),
    )


class RSSM(nn.Module):
    """PlaNet/Dreamer-style RSSM with deterministic and stochastic state."""

    def __init__(
        self,
        stoch_dim: int = 32,
        deter_dim: int = 200,
        hidden_dim: int = 200,
        action_dim: int = 3,
        embed_dim: int = 1024,
        min_std: float = 0.1,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.stoch_dim = int(stoch_dim)
        self.deter_dim = int(deter_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.min_std = float(min_std)

        self.input_mlp = nn.Sequential(
            nn.Linear(self.stoch_dim + self.action_dim, self.hidden_dim),
            _activation(activation),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim),
        )

    @property
    def feature_dim(self) -> int:
        return self.stoch_dim + self.deter_dim

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def features(self, states: RSSMState) -> torch.Tensor:
        return torch.cat([states.stoch, states.deter], dim=-1)

    def _dist_params(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mean
        return mean + std * torch.randn_like(std)

    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor, deterministic: bool = False) -> RSSMState:
        hidden = self.input_mlp(torch.cat([prev_state.stoch, action], dim=-1))
        deter = self.gru(hidden, prev_state.deter)
        mean, std = self._dist_params(self.prior_net(deter))
        stoch = self._sample(mean, std, deterministic=deterministic)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.imagine_step(prev_state, action, deterministic=deterministic)
        mean, std = self._dist_params(self.post_net(torch.cat([prior.deter, embed], dim=-1)))
        stoch = self._sample(mean, std, deterministic=deterministic)
        posterior = RSSMState(mean=mean, std=std, stoch=stoch, deter=prior.deter)
        return posterior, prior

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[RSSMState, RSSMState]:
        if embeds.ndim != 3:
            raise ValueError(f"embeds must have shape [B,T,D], got {tuple(embeds.shape)}")
        batch, time = embeds.shape[:2]
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B,T-1,A] or [B,T,A], got {tuple(actions.shape)}")
        if actions.shape[1] == time - 1:
            first = torch.zeros(batch, 1, actions.shape[-1], dtype=actions.dtype, device=actions.device)
            prev_actions = torch.cat([first, actions], dim=1)
        elif actions.shape[1] == time:
            prev_actions = actions
        else:
            raise ValueError(f"actions time dimension {actions.shape[1]} incompatible with embeds time {time}")

        prev = self.initial(batch, embeds.device)
        posteriors: list[RSSMState] = []
        priors: list[RSSMState] = []
        for step in range(time):
            posterior, prior = self.observe_step(prev, prev_actions[:, step], embeds[:, step], deterministic=deterministic)
            posteriors.append(posterior)
            priors.append(prior)
            prev = posterior
        return _stack_states(posteriors), _stack_states(priors)

    def imagine(self, initial_state: RSSMState, actions: torch.Tensor, deterministic: bool = False) -> RSSMState:
        prev = initial_state
        priors: list[RSSMState] = []
        for step in range(actions.shape[1]):
            prev = self.imagine_step(prev, actions[:, step], deterministic=deterministic)
            priors.append(prev)
        return _stack_states(priors)


def kl_divergence(lhs: RSSMState, rhs: RSSMState) -> torch.Tensor:
    """KL(lhs || rhs) for diagonal Gaussian RSSM states, returning ``[B,T]``."""

    lhs_var = lhs.std.square()
    rhs_var = rhs.std.square()
    kl = torch.log(rhs.std / lhs.std) + (lhs_var + (lhs.mean - rhs.mean).square()) / (2.0 * rhs_var) - 0.5
    return kl.sum(dim=-1)


__all__ = ["RSSM", "RSSMState", "kl_divergence"]
