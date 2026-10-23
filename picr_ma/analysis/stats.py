"""Statistics with independent training runs (seeds) as the unit of analysis.

Confidence intervals: per-run quantities (self-play return, permutation
sensitivity) use a percentile bootstrap over seeds. Cross-play is a mean over
ordered pairs of *distinct* runs; pairs that share a run are not independent,
so it uses a cluster bootstrap: resample seeds with replacement, then recompute
the all-pairs mean over the resampled runs (a run drawn twice is never paired
with itself). Percentile intervals under-cover somewhat at 5-10 seeds, so they
are descriptive.

Hypothesis tests between two methods use a permutation test over run labels:
pool the runs of both methods, re-split them at random into groups of the
original sizes, and recompute every statistic (cross-play included, using the
full cross-method pairing matrix). This is exact under the null hypothesis that
the two methods produce exchangeable runs.
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


def group_statistics(matrix: np.ndarray, idx: np.ndarray, per_run: dict[str, np.ndarray]) -> dict[str, float]:
    """Self-play, cross-play, gap and per-run means for the runs ``idx`` of a pooled pairing matrix."""
    sub = matrix[np.ix_(idx, idx)]
    sp = float(np.diag(sub).mean())
    xp = float(sub[~np.eye(len(idx), dtype=bool)].mean())
    return {"sp": sp, "xp": xp, "gap": sp - xp, **{k: float(v[idx].mean()) for k, v in per_run.items()}}


def permutation_test(
    matrix: np.ndarray,
    per_run: dict[str, np.ndarray],
    n_a: int,
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided permutation p-values for (group A - group B) of every statistic.

    ``matrix`` is the square cross-play matrix over the pooled runs, with the
    ``n_a`` runs of A first; ``per_run`` maps names to per-run values in the same
    order. p = (1 + #{|perm diff| >= |observed diff|}) / (1 + n_perm).
    """
    n = len(matrix)
    a, b = np.arange(n_a), np.arange(n_a, n)
    stat_a, stat_b = group_statistics(matrix, a, per_run), group_statistics(matrix, b, per_run)
    observed = {k: stat_a[k] - stat_b[k] for k in stat_a}
    exceed = dict.fromkeys(observed, 0)
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        order = rng.permutation(n)
        pa, pb = group_statistics(matrix, order[:n_a], per_run), group_statistics(matrix, order[n_a:], per_run)
        for k in observed:
            exceed[k] += abs(pa[k] - pb[k]) >= abs(observed[k]) - 1e-12
    return {k: (1 + exceed[k]) / (1 + n_perm) for k in observed}


def difference(a: Estimate, b: Estimate) -> dict[str, float]:
    """a - b for estimates from independent seed populations.

    Returns the point difference and a percentile 95% CI from the two
    independent bootstrap distributions. Use ``permutation_test`` for p-values.
    """
    lo, hi = np.nanpercentile(a.samples - b.samples, [2.5, 97.5])
    return {"diff": a.mean - b.mean, "ci_lo": float(lo), "ci_hi": float(hi)}
