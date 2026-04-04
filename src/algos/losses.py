from __future__ import annotations

import numpy as np


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if len(advantages) == 0:
        return advantages
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / max(std, eps)


def clipped_ppo_dlogp(
    ratios: np.ndarray,
    advantages: np.ndarray,
    clip_ratio: float,
) -> np.ndarray:
    coeff = np.zeros_like(advantages, dtype=np.float32)
    upper = 1.0 + clip_ratio
    lower = 1.0 - clip_ratio
    for index, (ratio, advantage) in enumerate(zip(ratios, advantages)):
        if advantage >= 0.0 and ratio > upper:
            coeff[index] = 0.0
        elif advantage < 0.0 and ratio < lower:
            coeff[index] = 0.0
        else:
            coeff[index] = -(advantage * ratio)
    return coeff
