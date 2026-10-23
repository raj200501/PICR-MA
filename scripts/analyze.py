#!/usr/bin/env python
"""Compute every number, table and figure in the paper from results/.

    python scripts/analyze.py

Reads results/main (and results/lambda if present) and writes
paper/figures/*.pdf, paper/tables/*.tex and results/analysis.json. Fails if the
runs on disk differ from the runs configs/experiments.yaml defines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis import figures, tables
from picr_ma.analysis.additive import additive_decomposition
from picr_ma.analysis.aggregate import (
    METHOD_ORDER,
    compare_to,
    crossplay_block,
    load_crossplay,
    load_curves,
    load_order_shift,
    load_summaries,
    method_estimates,
    method_matrix,
)
from picr_ma.analysis.stats import bootstrap_mean, bootstrap_pairs
from picr_ma.experiments import expand_suite, load_suites

ROOT = Path(__file__).resolve().parents[1]


def check_complete(suite: str, results_dir: Path, config_path: Path) -> None:
    """The finished runs on disk must be exactly the suite's runs, each trained with its current config."""
    runs = expand_suite(load_suites(config_path), suite, results_dir)
    unfinished = sorted(str(run.out_dir) for run in runs if not run.done)  # raises on a config mismatch
    unexpected = sorted(
        str(p.parent) for p in (results_dir / suite).glob("*/*/seed*/summary.json") if p.parent not in {r.out_dir for r in runs}
    )
    if unfinished or unexpected:
        raise SystemExit(
            f"results/{suite} does not match the config.\n  unfinished: {unfinished[:5]}\n  unexpected: {unexpected[:5]}"
        )


def within_method_spearman(per_run: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation between a run's permutation sensitivity and its cross-play, within each method."""
    out = {}
    for method, runs in per_run.groupby("method"):
        if runs.perm_js.std() > 0:
            out[method] = float(runs[["perm_js", "xp"]].corr("spearman").iloc[0, 1])
    return out


def per_run_table(summaries: pd.DataFrame, xp: pd.DataFrame, env: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        block = crossplay_block(xp, method, method)
        runs = summaries[(summaries.env == env) & (summaries.variant == method)].sort_values("seed")
        for i, (_, run) in enumerate(runs.iterrows()):
            others = np.delete(block[i], i)
            rows.append({"method": method, "seed": run.seed, "sp": block[i, i], "xp": others.mean(), "perm_js": run.perm_js})
    return pd.DataFrame(rows)


def lambda_table(suite_dir: Path) -> pd.DataFrame:
    summaries = load_summaries(suite_dir)
    rows = []
    for env in sorted(set(summaries.env)):
        xp = load_crossplay(suite_dir, env)
        for variant in sorted(set(summaries[summaries.env == env].variant)):
            lam = float(variant.split("_lam")[1])
            block = crossplay_block(xp, variant, variant)
            runs = summaries[(summaries.env == env) & (summaries.variant == variant)]
            sp, cross, js = bootstrap_mean(np.diag(block)), bootstrap_pairs(block, True), bootstrap_mean(runs.perm_js)
            rows.append(
                {
                    "env": env,
                    "lam": lam,
                    "n_seeds": len(runs),
                    **{f"sp{s}": v for s, v in zip(("", "_lo", "_hi"), (sp.mean, sp.lo, sp.hi))},
                    **{f"xp{s}": v for s, v in zip(("", "_lo", "_hi"), (cross.mean, cross.lo, cross.hi))},
                    **{f"perm_js{s}": v for s, v in zip(("", "_lo", "_hi"), (js.mean, js.lo, js.hi))},
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiments.yaml")
    args = parser.parse_args()
    fig_dir, tab_dir = args.paper_dir / "figures", args.paper_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    main_dir = args.results_dir / "main"
    check_complete("main", args.results_dir, args.config)
    summaries = load_summaries(main_dir)
    envs = [e for e in ("simple_spread_v3", "simple_adversary_v3") if e in set(summaries.env)]
    estimates, comparisons, curves, per_run, matrices, crossplays = {}, {}, {}, {}, {}, {}
    for env in envs:
        xp = load_crossplay(main_dir, env)
        crossplays[env] = xp
        shift = load_order_shift(main_dir, env)
        estimates[env] = method_estimates(summaries, xp, env, shift=shift)
        comparisons[env] = compare_to(estimates[env], summaries, xp, env, shift=shift)
        curves[env] = load_curves(main_dir, env)
        per_run[env] = per_run_table(summaries, xp, env)
        matrices[env] = method_matrix(xp)

    figures.learning_curves(curves, fig_dir / "learning_curves.pdf")
    figures.sp_vs_xp(estimates, fig_dir / "self_play_vs_cross_play.pdf")
    figures.sensitivity_vs_crossplay(per_run, fig_dir / "sensitivity_vs_cross_play.pdf")
    figures.additive_fit(crossplays, fig_dir / "additive_fit.pdf")
    (tab_dir / "main_results.tex").write_text(tables.main_table(estimates))
    (tab_dir / "comparison.tex").write_text(tables.comparison_table(comparisons))

    report = {
        "seeds_per_method": {env: summaries[summaries.env == env].groupby("variant").seed.nunique().to_dict() for env in envs},
        "estimates": {env: {m: {k: v.as_dict() for k, v in est.items()} for m, est in estimates[env].items()} for env in envs},
        "vs_ippo": comparisons,
        "method_crossplay_matrix_ego_by_partner": {env: matrices[env].round(3).to_dict(orient="index") for env in envs},
        "within_method_spearman_perm_js_vs_xp": {env: within_method_spearman(per_run[env]) for env in envs},
        "additive_decomposition": {env: additive_decomposition(crossplays[env]) for env in envs},
    }
    selection = args.results_dir / "lambda" / "selection.json"
    if selection.exists():
        report["selected_lambda"] = {env: v["selected_lambda"] for env, v in json.loads(selection.read_text()).items()}
    for name in ("reference_returns", "overhead"):
        path = args.results_dir / f"{name}.json"
        if path.exists():
            report[name] = json.loads(path.read_text())
    if "overhead" in report:
        (tab_dir / "overhead.tex").write_text(tables.overhead_table(report["overhead"]))
    lambda_dir = args.results_dir / "lambda"
    if lambda_dir.exists() and any(lambda_dir.glob("*/crossplay.csv")):
        check_complete("lambda", args.results_dir, args.config)
        sweep = lambda_table(lambda_dir)
        figures.lambda_sweep(sweep, fig_dir / "lambda_sweep.pdf")
        report["lambda_sweep"] = sweep.round(5).to_dict(orient="records")

    (tab_dir / "numbers.tex").write_text(tables.numbers_tex(estimates, comparisons, report))
    (args.results_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    (args.results_dir / "main_results.md").write_text(tables.markdown_table(estimates))
    print(tables.markdown_table(estimates))


if __name__ == "__main__":
    main()
