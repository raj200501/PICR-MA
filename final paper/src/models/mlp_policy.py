from __future__ import annotations

from src.models.actor_critic import MLPActorCritic


def build_mlp_policy(
    obs_dim: int,
    act_dim: int,
    seed: int = 0,
    hidden_dim: int = 64,
) -> MLPActorCritic:
    return MLPActorCritic(
        input_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
