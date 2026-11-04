"""Load a suite's run artifacts and compute the quantities reported in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis.stats import Estimate, bootstrap_mean, bootstrap_pairs, difference, permutation_test, resample_runs

METHOD_ORDER = ("ippo", "picr", "deepsets", "order_aug", "order_rand")
METHOD_LABELS = {"ippo": "IPPO", "picr": "PICR", "deepsets": "Deep Sets", "order_aug": "Order aug.", "order_rand": "Order rand."}
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


def crossplay_block(xp: pd.DataFrame, ego: str, partner: str, value: str = "return_mean") -> np.ndarray:
    """(n_ego_seeds, n_partner_seeds) matrix of a cross-play column, seeds sorted."""
    block = xp[(xp.ego_method == ego) & (xp.partner_method == partner)]
    return block.pivot(index="ego_seed", columns="partner_seed", values=value).sort_index().sort_index(axis=1).to_numpy()


def slot_contrast_block(xp: pd.DataFrame, ego: str, partner: str) -> np.ndarray:
    """Cross-play with the ego in slot 0 minus with the ego in the last slot.

    MPE lists teammates in index order, so in slot 0 both partners list the ego
    as their first teammate and in the last slot both list it second. The two
    seatings differ only by teammate order (reversing every agent's list turns
    one into the other), so the contrast is zero in expectation unless the
    policies' returns depend on teammate order.
    """
    last = max(int(c.removeprefix("return_mean_slot")) for c in xp.columns if c.startswith("return_mean_slot"))
    return crossplay_block(xp, ego, partner, "return_mean_slot0") - crossplay_block(xp, ego, partner, f"return_mean_slot{last}")


def load_order_shift(suite_dir: str | Path, env: str) -> pd.DataFrame | None:
    path = Path(suite_dir) / env / "order_shift.csv"
    return pd.read_csv(path) if path.exists() else None


SHIFT_METRICS = ("shift_drop", "relabel_drop", "resample_drop")


def order_shift_per_run(shift: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per run (sorted by seed): self-play return lost under each kind of teammate-order condition.

    shift_drop     genuine order shifts (views no relabeling of the agents produces)
    relabel_drop   relabeling patterns: zero in expectation for any shared policy, a noise floor
    resample_drop  teammate features replaced with another episode's: reliance on teammates at all
    """
    rows = shift[shift.method == method]
    by_kind = rows.groupby(["seed", "kind"]).return_mean.mean().unstack()
    out = pd.DataFrame({"sp": by_kind["identity"]}).sort_index()
    out["shift_drop"] = out.sp - by_kind["genuine"]
    out["relabel_drop"] = out.sp - by_kind["relabeling"]
    out["resample_drop"] = out.sp - by_kind["resampled"]
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
    gap      sp - xp, bootstrapped jointly over seeds (the primary endpoint:
             individual competence cancels in the difference)
    slot_contrast  cross-play with the ego listed first by both partners minus
             listed second (see ``slot_contrast_block``); zero unless order matters
    perm_js  permutation sensitivity on self-play evaluation states
    flip     fraction of (state, permutation) pairs whose greedy action changes
    shift_drop, relabel_drop, resample_drop
             (with ``shift``) self-play return lost under genuine order shifts,
             under relabeling patterns (a null control) and with teammate
             features resampled; see ``order_shift_per_run``
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
        if any(c.startswith("return_mean_slot") for c in xp.columns):
            out[method]["slot_contrast"] = bootstrap_pairs(slot_contrast_block(xp, method, method), same_population=True, seed=i)
        if shift is not None:
            per_run = order_shift_per_run(shift, method)
            for key in SHIFT_METRICS:
                out[method][key] = bootstrap_mean(per_run[key].to_numpy(), seed=i)
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


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests."""
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def compare_to(
    estimates: dict[str, dict[str, Estimate]],
    summaries: pd.DataFrame,
    xp: pd.DataFrame,
    env: str,
    baseline: str = "ippo",
    n_perm: int = 10_000,
    shift: pd.DataFrame | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Each method minus the baseline, for every metric.

    diff, ci_lo, ci_hi  point difference and bootstrap 95% interval
    mde                 minimum detectable effect, 2.8 x the bootstrap SD of the
                        difference (80% power, two-sided alpha = 0.05)
    p                   permutation-test p-value over run labels
    p_holm              p adjusted with Holm's method across the compared methods
    """
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
            for key in SHIFT_METRICS:
                per_run[key] = np.concatenate([x[key].to_numpy() for x in shifts])
        pair_stats = None
        if "slot_contrast" in metrics:
            pair_stats = {
                "slot_contrast": np.block([[slot_contrast_block(xp, ego, partner) for partner in order] for ego in order])
            }
        n_a = int((runs.variant == method).sum())
        p_values = permutation_test(matrix, per_run, n_a, n_perm=n_perm, pair_stats=pair_stats)
        out[method] = {}
        for name in metrics:
            if name not in p_values:
                continue
            diff = difference(metrics[name], estimates[baseline][name])
            out[method][name] = {**diff, "mde": 2.8 * diff["sd"], "p": p_values.get(name, float("nan"))}
    for name in {n for comp in out.values() for n in comp}:
        family = {m: comp[name]["p"] for m, comp in out.items() if name in comp and not np.isnan(comp[name]["p"])}
        for m, p in holm(family).items():
            out[m][name]["p_holm"] = p
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
