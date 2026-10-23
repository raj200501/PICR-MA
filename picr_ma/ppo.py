"""Clipped PPO with GAE for a shared-parameter team (IPPO)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from picr_ma.envs.slicing import TeammateSlicing
from picr_ma.networks import ActorCritic
from picr_ma.regularizers import picr_loss


@dataclass
class PPOConfig:
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    picr_lambda: float = 0.0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE along the time axis (axis 1) for arrays shaped (episodes, T, agents).

    Each slice ``[:, :, i]`` is one agent's own trajectory, so agents never
    bootstrap from each other's values. Episodes end at T: MPE episodes have a
    fixed horizon and the objective is the undiscounted-horizon return, so the
    value after the last step is taken to be zero.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[::2], dtype=np.float32)
    next_value = np.zeros_like(last)
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last = delta + gamma * gae_lambda * last
        advantages[:, t] = last
        next_value = values[:, t]
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    config: PPOConfig,
    rng: np.random.Generator,
    slicing: TeammateSlicing | None = None,
) -> dict[str, float]:
    """Run ``config.epochs`` epochs of minibatch PPO on a flat batch.

    ``batch`` holds flat arrays: obs, actions, log_probs, advantages, returns.
    When ``config.picr_lambda > 0`` the PICR penalty is added on the same
    minibatch observations.
    """
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32)

    use_picr = config.picr_lambda > 0 and slicing is not None and slicing.n_teammates > 1
    n = len(obs)
    mb_size = max(1, n // config.minibatches)
    totals = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "picr_loss")}
    count = 0
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for start in range(0, n - mb_size + 1, mb_size):
            idx = torch.as_tensor(order[start : start + mb_size])
            mb_adv = advantages[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            logits = model.logits(obs[idx])
            log_probs_all = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs_all.gather(1, actions[idx, None]).squeeze(-1)
            entropy = -(log_probs_all.exp() * log_probs_all).sum(-1).mean()
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * mb_adv).mean()
            value_loss = (model.value(obs[idx]) - returns[idx]).pow(2).mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            reg = picr_loss(model.logits, obs[idx], slicing, rng) if use_picr else torch.zeros(())
            loss = loss + config.picr_lambda * reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                totals["clip_frac"] += ((ratio - 1).abs() > config.clip).float().mean().item()
                totals["picr_loss"] += reg.item()
            count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}
