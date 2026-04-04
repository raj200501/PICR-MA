from __future__ import annotations

import numpy as np

from src.envs.observation_slicing import ObservationSlicing, permute_teammates


def role_permutation(rng: np.random.Generator, n_agents: int) -> tuple[int, ...]:
    return tuple(rng.permutation(np.arange(n_agents)).tolist())


def permute_observation_batch(
    observations: np.ndarray,
    slicing: ObservationSlicing,
    rng: np.random.Generator,
) -> np.ndarray:
    if slicing.n_teammates <= 1:
        return np.asarray(observations, dtype=np.float32)
    permutation = tuple(rng.permutation(np.arange(slicing.n_teammates)).tolist())
    return np.stack(
        [permute_teammates(obs, slicing, permutation) for obs in observations],
        axis=0,
    )
