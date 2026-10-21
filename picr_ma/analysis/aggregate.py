"""Load a suite's run artifacts and compute the quantities reported in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis.stats import Estimate, bootstrap_mean, bootstrap_pairs, difference

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


def method_estimates(summaries: pd.DataFrame, xp: pd.DataFrame, env: str, methods=METHOD_ORDER) -> dict[str, dict[str, Estimate]]:
    """Per-method estimates for one environment.

    sp       self-play return (diagonal of the cross-play matrix: same seeds,
             same episodes as cross-play)
    xp       cross-play return with independently trained partners of the same method
    gap      sp - xp, bootstrapped jointly over seeds
    perm_js  permutation sensitivity on self-play evaluation states
    flip     fraction of (state, permutation) pairs whose greedy action changes
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
            "sec_per_1k_steps": bootstrap_mean(1000 * runs.train_seconds.to_numpy() / runs.env_steps.to_numpy(), seed=i),
        }
    return out


def _gap_samples(block: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap samples of (self-play mean - cross-play mean) using one seed resample for both."""
    rng = np.random.default_rng(seed)
    n = len(block)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = block[np.ix_(idx, idx)]
        off = np.where(idx[:, None] == idx[None, :], np.nan, sub)
        samples[b] = np.diag(sub).mean() - np.nanmean(off)
    return samples


def compare_to(estimates: dict[str, dict[str, Estimate]], baseline: str = "ippo") -> dict[str, dict[str, dict[str, float]]]:
    out = {}
    for method, metrics in estimates.items():
        if method == baseline:
            continue
        out[method] = {name: difference(metrics[name], estimates[baseline][name]) for name in metrics}
    return out


def method_matrix(xp: pd.DataFrame, methods=METHOD_ORDER) -> pd.DataFrame:
    """Mean cross-play between methods (rows: ego, columns: partner), self-pairs excluded."""
    present = [m for m in methods if m in set(xp.ego_method)]
    table = pd.DataFrame(index=present, columns=present, dtype=float)
    for ego in present:
        for partner in present:
            block = crossplay_block(xp, ego, partner)
            same = ego == partner
            table.loc[ego, partner] = bootstrap_pairs(block, same_population=same, n_boot=1).mean
    return table
