from __future__ import annotations

import numpy as np

from picr_ma.envs.observation_slicing import ObservationSlicing, permute_teammates
from picr_ma.methods.picr_ma import all_permutations
from picr_ma.utils.metrics_common import js_divergence


def sensitivity(
    policy_probs_fn,
    obs: np.ndarray,
    slicing: ObservationSlicing,
    permutations: list[tuple[int, ...]] | None = None,
) -> tuple[float, list[float]]:
    original = policy_probs_fn(np.asarray([obs], dtype=np.float32))[0]
    values = []
    perms = permutations if permutations is not None else all_permutations(slicing)
    for permutation in perms:
        permuted_obs = permute_teammates(obs, slicing, permutation)
        permuted = policy_probs_fn(np.asarray([permuted_obs], dtype=np.float32))[0]
        values.append(js_divergence(original, permuted))
    if not values:
        return 0.0, [0.0]
    return float(np.mean(values)), values
