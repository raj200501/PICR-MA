from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.algos.losses import clipped_ppo_dlogp, normalize_advantages
from src.methods.picr_ma import build_picr_views, symmetric_kl_with_gradients


@dataclass
class PPOConfig:
    learning_rate: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    update_epochs: int = 4
    picr_lambda: float = 0.0
    teammate_dropout_p: float = 0.0


def compute_returns_and_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * mask - values[index]
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]
    returns = advantages + values
    return returns, advantages


def ppo_update(
    policy,
    batch: dict[str, np.ndarray],
    rng: np.random.Generator,
    config: PPOConfig,
    slicing=None,
) -> dict[str, float]:
    observations = batch["observations"]
    raw_observations = batch["raw_observations"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    returns = batch["returns"]
    advantages = normalize_advantages(batch["advantages"])
    action_count = policy.action_dim

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "picr_loss": 0.0,
    }

    for _ in range(config.update_epochs):
        forward = policy.forward(observations)
        probs = forward.probs
        selected_probs = probs[np.arange(len(actions)), actions]
        current_log_probs = np.log(selected_probs + 1e-8)
        ratios = np.exp(current_log_probs - old_log_probs)
        dlogp = clipped_ppo_dlogp(ratios, advantages, config.clip_ratio) / len(actions)

        one_hot = np.eye(action_count, dtype=np.float32)[actions]
        dlogits = dlogp[:, None] * (probs - one_hot)

        value_errors = (forward.values - returns).astype(np.float32)
        dvalues = config.value_coef * 2.0 * value_errors / len(actions)

        gradients = policy.policy_value_backward(forward, dlogits, dvalues)
        stats["policy_loss"] += float(-np.mean(np.minimum(ratios * advantages, np.clip(ratios, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages)))
        stats["value_loss"] += float(np.mean(value_errors**2))

        if config.picr_lambda > 0.0 and slicing is not None and slicing.n_teammates > 1:
            view_left, view_right = build_picr_views(
                raw_observations,
                slicing=slicing,
                rng=rng,
                dropout_p=config.teammate_dropout_p,
            )
            forward_left = policy.forward(view_left)
            forward_right = policy.forward(view_right)
            picr_loss, dlogits_left, dlogits_right = symmetric_kl_with_gradients(
                forward_left.probs,
                forward_right.probs,
            )
            grads_left = policy.policy_value_backward(
                forward_left,
                config.picr_lambda * dlogits_left,
                np.zeros(len(view_left), dtype=np.float32),
            )
            grads_right = policy.policy_value_backward(
                forward_right,
                config.picr_lambda * dlogits_right,
                np.zeros(len(view_right), dtype=np.float32),
            )
            for name in gradients:
                gradients[name] += grads_left[name] + grads_right[name]
            stats["picr_loss"] += picr_loss

        policy.apply_gradients(gradients, config.learning_rate)

    for key in stats:
        stats[key] /= config.update_epochs
    return stats
