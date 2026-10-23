"""Load a suite's run artifacts and compute the quantities reported in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis.stats import Estimate, bootstrap_mean, bootstrap_pairs, difference, permutation_test, resample_runs

METHOD_ORDER = ("ippo", "picr", "deepsets", "order_aug")
METHOD_LABELS = {"ippo": "IPPO", "picr": "PICR", "deepsets": "Deep Sets", "order_aug": "Order aug."}
ENV_LABELS = {"simple_spread_v3": "Spread", "simple_adversary_v3": "Adversary"}


def load_summaries(suite_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(suite_dir).glob("*/*/seed*/summary.json")):
        summary = json.loads(path.read_text())
        summary["variant"] = path.parent.parent.name
        rows.append(summary)
    return pd.DataFrame(rows)


def load_curves(suite_dir: str | Path, env: str) -> pd.DataFrame:
    frames = []
    for path in sorted((Path(suite_dir) / env).glob("*/seed*/metrics.csv.gz")):
        frame = pd.read_csv(path)
        frame["variant"] = path.parent.parent.name
        frame["seed"] = int(path.parent.name.removeprefix("seed"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_crossplay(suite_dir: str | Path, env: str) -> pd.DataFrame:
    return pd.read_csv(Path(suite_dir) / env / "crossplay.csv")


def crossplay_block(xp: pd.DataFrame, ego: str, partner: str) -> np.ndarray:
    """(n_ego_seeds, n_partner_seeds) matrix of mean cross-play returns, seeds sorted."""
    block = xp[(xp.ego_method == ego) & (xp.partner_method == partner)]
    return block.pivot(index="ego_seed", columns="partner_seed", values="return_mean").sort_index().sort_index(axis=1).to_numpy()


def load_order_shift(suite_dir: str | Path, env: str) -> pd.DataFrame | None:
    path = Path(suite_dir) / env / "order_shift.csv"
    return pd.read_csv(path) if path.exists() else None


def order_shift_per_run(shift: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per run (sorted by seed): self-play under identity order, under shifted orders, and with teammates ablated."""
    rows = shift[shift.method == method]
    by_kind = lambda kind: rows[rows.kind == kind].groupby("seed").return_mean.mean()  # noqa: E731
    out = pd.DataFrame(
        {"sp": by_kind("identity"), "sp_shift": by_kind("permuted"), "sp_ablated": by_kind("ablated")}
    ).sort_index()
    out["shift_drop"] = out.sp - out.sp_shift
    out["ablation_drop"] = out.sp - out.sp_ablated
    return out


def method_estimates(
    summaries: pd.DataFrame,
    xp: pd.DataFrame,
    env: str,
    methods=METHOD_ORDER,
    shift: pd.DataFrame | None = None,
) -> dict[str, dict[str, Estimate]]:
    """Per-method estimates for one environment.

    sp       self-play return (diagonal of the cross-play matrix: same seeds,
             same episodes as cross-play)
    xp       cross-play return with independently trained partners of the same method
    gap      sp - xp, bootstrapped jointly over seeds
    perm_js  permutation sensitivity on self-play evaluation states
    flip     fraction of (state, permutation) pairs whose greedy action changes
    shift_drop     (with ``shift``) self-play minus self-play when every agent sees
                   its teammates in a different order; paired over the same episodes
    ablation_drop  (with ``shift``) self-play minus self-play with all teammate
                   features zeroed: a control for how much teammates matter at all
    """
    out = {}
    for i, method in enumerate(methods):
        block = crossplay_block(xp, method, method)
        if block.size == 0:
            continue
        runs = summaries[(summaries.env == env) & (summaries.variant == method)].sort_values("seed")
        sp = bootstrap_mean(np.diag(block), seed=i)
        cross = bootstrap_pairs(block, same_population=True, seed=i)
        gap_samples = _gap_samples(block, seed=i)
        out[method] = {
            "sp": sp,
            "xp": cross,
            "gap": Estimate(sp.mean - cross.mean, *np.percentile(gap_samples, [2.5, 97.5]), gap_samples),
            "perm_js": bootstrap_mean(runs.perm_js.to_numpy(), seed=i),
            "flip": bootstrap_mean(runs.perm_flip_rate.to_numpy(), seed=i),
        }
        if shift is not None:
            per_run = order_shift_per_run(shift, method)
            out[method]["shift_drop"] = bootstrap_mean(per_run.shift_drop.to_numpy(), seed=i)
            out[method]["ablation_drop"] = bootstrap_mean(per_run.ablation_drop.to_numpy(), seed=i)
    return out


def _gap_samples(block: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap samples of (self-play mean - cross-play mean) using one seed resample for both."""
    rng = np.random.default_rng(seed)
    n = len(block)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = resample_runs(n, rng, need_distinct=True)
        sub = block[np.ix_(idx, idx)]
        off = np.where(idx[:, None] == idx[None, :], np.nan, sub)
        samples[b] = np.diag(sub).mean() - np.nanmean(off)
    return samples


def compare_to(
    estimates: dict[str, dict[str, Estimate]],
    summaries: pd.DataFrame,
    xp: pd.DataFrame,
    env: str,
    baseline: str = "ippo",
    n_perm: int = 10_000,
    shift: pd.DataFrame | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Each method minus the baseline: point difference, bootstrap 95% CI and permutation p-value."""
    out = {}
    for method, metrics in estimates.items():
        if method == baseline:
            continue
        order = [method, baseline]
        matrix = np.block([[crossplay_block(xp, ego, partner) for partner in order] for ego in order])
        runs = pd.concat([summaries[(summaries.env == env) & (summaries.variant == m)].sort_values("seed") for m in order])
        per_run = {"perm_js": runs.perm_js.to_numpy(), "flip": runs.perm_flip_rate.to_numpy()}
        if shift is not None:
            shifts = [order_shift_per_run(shift, m) for m in order]
            per_run["shift_drop"] = np.concatenate([x.shift_drop.to_numpy() for x in shifts])
            per_run["ablation_drop"] = np.concatenate([x.ablation_drop.to_numpy() for x in shifts])
        n_a = int((runs.variant == method).sum())
        p_values = permutation_test(matrix, per_run, n_a, n_perm=n_perm)
        out[method] = {name: {**difference(metrics[name], estimates[baseline][name]), "p": p_values[name]} for name in metrics}
    return out


def method_matrix(xp: pd.DataFrame, methods=METHOD_ORDER) -> pd.DataFrame:
    """Mean cross-play between methods (index: ego method, columns: partner method), self-pairs excluded."""
    present = [m for m in methods if m in set(xp.ego_method)]
    table = pd.DataFrame(index=present, columns=present, dtype=float)
    for ego in present:
        for partner in present:
            block = crossplay_block(xp, ego, partner)
            same = ego == partner
            table.loc[ego, partner] = bootstrap_pairs(block, same_population=same, n_boot=1).mean
    return table
