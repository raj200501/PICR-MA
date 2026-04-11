from __future__ import annotations

import numpy as np

from src.envs.observation_slicing import ObservationSlicing, permute_teammates


def role_permutation(rng: np.random.Generator, n_agents: int) -> tuple[int, ...]:
    return tuple(rng.permutation(np.arange(n_agents)).tolist())


def teammate_permutation(
    rng: np.random.Generator,
    n_teammates: int,
) -> tuple[int, ...]:
    return tuple(rng.permutation(np.arange(n_teammates)).tolist())


def permute_observation_batch(
    observations: np.ndarray,
    slicing: ObservationSlicing,
    rng: np.random.Generator,
    permutation: tuple[int, ...] | None = None,
) -> np.ndarray:
    if slicing.n_teammates <= 1:
        return np.asarray(observations, dtype=np.float32)
    chosen_permutation = permutation or teammate_permutation(rng, slicing.n_teammates)
    return np.stack(
        [permute_teammates(obs, slicing, chosen_permutation) for obs in observations],
        axis=0,
    )
