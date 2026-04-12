from __future__ import annotations

import math

import numpy as np


def mean_se(values) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if len(arr) == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    se = float(arr.std(ddof=0) / math.sqrt(len(arr)))
    return mean, se


def bootstrap_mean_ci(
    values,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if len(arr) == 1 or np.allclose(arr, arr[0]):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float32)
    for index in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot[index] = sample.mean()
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return mean, lower, upper
