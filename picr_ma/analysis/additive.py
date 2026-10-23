"""Is cross-play explained by individual competence, or by pair-specific compatibility?

Model the return of a team with ego run i and partner run j (i != j) as

    R_ij = mu + a_i + b_j + e_ij

where a_i and b_j are how good run i is as an ego and run j as a partner. If
runs follow incompatible conventions, some pairs do much worse than their
individual competences predict, which shows up as structured residuals e_ij.
We report the variance explained by the additive part and the split-half
reliability of the residuals: if the residuals estimated from two disjoint
halves of the evaluation episodes are uncorrelated, pair-specific
compatibility effects are indistinguishable from evaluation noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_matrix(xp: pd.DataFrame, value: str = "return_mean") -> tuple[np.ndarray, list[str]]:
    """Square matrix over all runs (rows: ego, columns: partner) of ``value``."""
    runs = sorted(set(xp.ego_method + "/" + xp.ego_seed.astype(str)))
    frame = xp.assign(
        ego=xp.ego_method + "/" + xp.ego_seed.astype(str), partner=xp.partner_method + "/" + xp.partner_seed.astype(str)
    )
    return frame.pivot(index="ego", columns="partner", values=value).loc[runs, runs].to_numpy(), runs


def fit_additive(matrix: np.ndarray, iterations: int = 500) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares fit of mu + a_i + b_j to the off-diagonal entries (alternating updates)."""
    n = len(matrix)
    mask = ~np.eye(n, dtype=bool)
    mu = matrix[mask].mean()
    a, b = np.zeros(n), np.zeros(n)
    for _ in range(iterations):
        a = np.array([(matrix[i] - mu - b)[mask[i]].mean() for i in range(n)])
        b = np.array([(matrix[:, j] - mu - a)[mask[:, j]].mean() for j in range(n)])
        shift = a.mean()
        a, mu = a - shift, mu + shift
    return mu, a, b


def residuals(matrix: np.ndarray) -> np.ndarray:
    mu, a, b = fit_additive(matrix)
    mask = ~np.eye(len(matrix), dtype=bool)
    return (matrix - (mu + a[:, None] + b[None, :]))[mask]


def additive_decomposition(xp: pd.DataFrame) -> dict[str, float]:
    full, _ = pair_matrix(xp)
    mask = ~np.eye(len(full), dtype=bool)
    mu, a, b = fit_additive(full)
    res = (full - (mu + a[:, None] + b[None, :]))[mask]
    total = full[mask] - full[mask].mean()
    out = {
        "n_runs": len(full),
        "r2_additive": float(1 - (res**2).sum() / (total**2).sum()),
        "sd_total": float(total.std()),
        "sd_residual": float(res.std()),
        "sd_ego_effect": float(a.std()),
        "sd_partner_effect": float(b.std()),
        "corr_ego_partner_effect": float(np.corrcoef(a, b)[0, 1]),
    }
    if {"return_mean_half1", "return_mean_half2"} <= set(xp.columns):
        r1, r2 = residuals(pair_matrix(xp, "return_mean_half1")[0]), residuals(pair_matrix(xp, "return_mean_half2")[0])
        out["residual_split_half_corr"] = float(np.corrcoef(r1, r2)[0, 1])
    return out
