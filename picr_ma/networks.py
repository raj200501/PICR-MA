"""Actor-critic networks.

Two encoders share one interface:

* ``mlp``      - a plain MLP over the flat observation. Its output depends on
                 teammate order (the baseline that PICR regularizes).
* ``deepsets`` - teammate blocks pass through a shared encoder and are
                 mean-pooled, so the output is invariant to teammate order by
                 construction (Zaheer et al., 2017).

Actor and critic use separate encoders, as is standard for PPO on MPE.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from picr_ma.envs.slicing import TeammateSlicing, split_self_teammates


def _layer(n_in: int, n_out: int, gain: float = float(np.sqrt(2))) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DeepSetsEncoder(nn.Module):
    def __init__(self, slicing: TeammateSlicing, hidden: int) -> None:
        super().__init__()
        self.slicing = slicing
        self.phi = nn.Sequential(_layer(slicing.block_dim, hidden), nn.Tanh())
        self.rho = nn.Sequential(_layer(len(slicing.self_indices) + hidden, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self_feats, teammates = split_self_teammates(obs, self.slicing)
        pooled = self.phi(teammates).mean(dim=1)
        return self.rho(torch.cat([self_feats, pooled], dim=-1))


def make_encoder(kind: str, obs_dim: int, hidden: int, slicing: TeammateSlicing | None) -> nn.Module:
    if kind == "mlp":
        return MLPEncoder(obs_dim, hidden)
    if kind == "deepsets":
        if slicing is None:
            raise ValueError("deepsets encoder needs a TeammateSlicing")
        return DeepSetsEncoder(slicing, hidden)
    raise KeyError(f"Unknown encoder {kind!r}")


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        encoder: str = "mlp",
        hidden: int = 64,
        slicing: TeammateSlicing | None = None,
    ) -> None:
        super().__init__()
        self.config = {"obs_dim": obs_dim, "n_actions": n_actions, "encoder": encoder, "hidden": hidden}
        self.actor_body = make_encoder(encoder, obs_dim, hidden, slicing)
        self.critic_body = make_encoder(encoder, obs_dim, hidden, slicing)
        self.policy_head = _layer(hidden, n_actions, gain=0.01)
        self.value_head = _layer(hidden, 1, gain=1.0)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.actor_body(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_body(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, generator: torch.Generator | None = None, greedy: bool = False):
        """Sample actions for a batch of observations. Returns (actions, log_probs, values)."""
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.logits(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        if greedy:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
        chosen = log_probs.gather(1, actions[:, None]).squeeze(-1)
        return actions.numpy(), chosen.numpy(), self.value(x).numpy()

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return torch.softmax(self.logits(torch.as_tensor(obs, dtype=torch.float32)), dim=-1).numpy()


def save_policy(model: ActorCritic, path, slicing: TeammateSlicing | None = None) -> None:
    torch.save(
        {
            "config": model.config,
            "slicing": None if slicing is None else (slicing.obs_dim, slicing.teammate_blocks),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_policy(path) -> ActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    slicing = None
    if payload["slicing"] is not None:
        obs_dim, blocks = payload["slicing"]
        slicing = TeammateSlicing(obs_dim=obs_dim, teammate_blocks=tuple(tuple(b) for b in blocks))
    model = ActorCritic(**payload["config"], slicing=slicing)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
