from __future__ import annotations

import itertools

import numpy as np

from src.envs.observation_slicing import ObservationSlicing, permute_teammates, zero_teammates


def _normalize(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), eps, None)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p_norm = _normalize(p, eps)
    q_norm = _normalize(q, eps)
    kl_pq = np.sum(p_norm * (np.log(p_norm) - np.log(q_norm)), axis=-1)
    kl_qp = np.sum(q_norm * (np.log(q_norm) - np.log(p_norm)), axis=-1)
    return float(np.mean(0.5 * (kl_pq + kl_qp)))


def teammate_dropout(
    observations: np.ndarray,
    slicing: ObservationSlicing,
    dropout_p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if dropout_p <= 0.0 or slicing.n_teammates == 0:
        return np.asarray(observations, dtype=np.float32)
    outputs = []
    for obs in np.asarray(observations, dtype=np.float32):
        zero_mask = rng.random(slicing.n_teammates) < dropout_p
        outputs.append(zero_teammates(obs, slicing, zero_mask))
    return np.stack(outputs, axis=0)


def sample_two_permutations(
    slicing: ObservationSlicing,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = list(range(slicing.n_teammates))
    if len(indices) <= 1:
        identity = tuple(indices)
        return identity, identity
    perm1 = tuple(rng.permutation(indices).tolist())
    perm2 = tuple(rng.permutation(indices).tolist())
    return perm1, perm2


def build_picr_views(
    observations: np.ndarray,
    slicing: ObservationSlicing,
    rng: np.random.Generator,
    dropout_p: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    obs_arr = teammate_dropout(observations, slicing, dropout_p, rng)
    left = []
    right = []
    for obs in obs_arr:
        perm1, perm2 = sample_two_permutations(slicing, rng)
        left.append(permute_teammates(obs, slicing, perm1))
        right.append(permute_teammates(obs, slicing, perm2))
    return np.stack(left, axis=0), np.stack(right, axis=0)


def symmetric_kl_with_gradients(
    probs_left: np.ndarray,
    probs_right: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray, np.ndarray]:
    p = _normalize(probs_left, eps)
    q = _normalize(probs_right, eps)
    loss = 0.5 * np.sum(p * (np.log(p) - np.log(q)), axis=1) + 0.5 * np.sum(
        q * (np.log(q) - np.log(p)),
        axis=1,
    )

    grad_p = 0.5 * (np.log(p) - np.log(q) + 1.0 - (q / p))
    grad_q = 0.5 * (np.log(q) - np.log(p) + 1.0 - (p / q))

    def softmax_pullback(probs: np.ndarray, upstream: np.ndarray) -> np.ndarray:
        centered = upstream - np.sum(upstream * probs, axis=1, keepdims=True)
        return probs * centered / probs.shape[0]

    return (
        float(np.mean(loss)),
        softmax_pullback(p, grad_p),
        softmax_pullback(q, grad_q),
    )


def all_permutations(slicing: ObservationSlicing) -> list[tuple[int, ...]]:
    if slicing.n_teammates <= 1:
        return [tuple(range(slicing.n_teammates))]
    return list(itertools.permutations(range(slicing.n_teammates)))
