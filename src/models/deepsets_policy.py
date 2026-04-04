from __future__ import annotations

from src.envs.observation_slicing import ObservationSlicing
from src.models.actor_critic import DeepSetsActorCritic


def build_deepsets_policy(
    obs_dim: int,
    act_dim: int,
    slicing: ObservationSlicing,
    seed: int = 0,
    hidden_dim: int = 64,
) -> DeepSetsActorCritic:
    return DeepSetsActorCritic(
        input_dim=obs_dim,
        action_dim=act_dim,
        slicing=slicing,
        hidden_dim=hidden_dim,
        seed=seed,
    )
