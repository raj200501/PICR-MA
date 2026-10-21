"""Bootstrap statistics with independent training seeds as the unit of analysis.

Per-run quantities (self-play return, permutation sensitivity) use an ordinary
bootstrap over seeds. Cross-play is a mean over ordered pairs of *distinct*
runs; pairs that share a run are not independent, so it uses a cluster
bootstrap: resample seeds with replacement, then recompute the all-pairs mean
over the resampled runs (a run drawn twice is never paired with itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BOOT = 10_000


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    samples: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "ci_lo": self.lo, "ci_hi": self.hi}


def _estimate(point: float, samples: np.ndarray, alpha: float = 0.05) -> Estimate:
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(point), float(lo), float(hi), samples)


def bootstrap_mean(values, n_boot: int = N_BOOT, seed: int = 0) -> Estimate:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return _estimate(values.mean(), values[idx].mean(axis=1))


def _pair_mean(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray, same_population: bool) -> float:
    block = matrix[np.ix_(rows, cols)]
    if same_population:
        block = np.where(rows[:, None] == cols[None, :], np.nan, block)
    return float(np.nanmean(block))


def bootstrap_pairs(
    matrix: np.ndarray,
    same_population: bool,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Estimate:
    """Mean of ``matrix[i, j]`` over distinct runs, with a cluster bootstrap CI.

    ``matrix`` is (n_ego_runs, n_partner_runs). With ``same_population`` the
    egos and partners are the same runs (rows and columns index the same
    seeds), the diagonal is excluded, and one resample of seeds is used for
    both axes.
    """
    n_rows, n_cols = matrix.shape
    rng = np.random.default_rng(seed)
    point = _pair_mean(matrix, np.arange(n_rows), np.arange(n_cols), same_population)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        rows = resample_runs(n_rows, rng, need_distinct=same_population)
        cols = rows if same_population else rng.integers(0, n_cols, n_cols)
        samples[b] = _pair_mean(matrix, rows, cols, same_population)
    return _estimate(point, samples)


def resample_runs(n: int, rng: np.random.Generator, need_distinct: bool = False) -> np.ndarray:
    """Bootstrap indices; with ``need_distinct``, redraw until at least two distinct runs are present.

    Within-population cross-play is undefined for a resample containing a single
    run (it has no distinct pairs), so such draws are rejected.
    """
    while True:
        idx = rng.integers(0, n, n)
        if not need_distinct or n < 2 or len(np.unique(idx)) > 1:
            return idx


def difference(a: Estimate, b: Estimate) -> dict[str, float]:
    """a - b for estimates from independent seed populations.

    Returns the point difference, a percentile 95% CI and a two-sided
    bootstrap p-value (twice the smaller tail mass at zero).
    """
    diff = a.samples - b.samples
    tail = min(np.mean(diff <= 0), np.mean(diff >= 0))
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return {"diff": a.mean - b.mean, "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(1.0, 2 * tail))}
