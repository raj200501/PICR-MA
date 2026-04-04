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
