#!/usr/bin/env python
"""Compute every number, table and figure in the paper from results/.

    python scripts/analyze.py

Reads the committed results/ (training runs, cross-play, order shift, the
lambda suite, reference returns, instrument checks and benchmarks) and writes
paper/figures/*.pdf, docs/figures/*.png, paper/tables/*.tex,
results/analysis.json and results/main_results.md. It trains and evaluates
nothing. Fails if an input is missing or if the runs on disk differ from the
runs configs/experiments.yaml defines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis import figures, tables
from picr_ma.analysis.additive import additive_decomposition, run_compatibility
from picr_ma.analysis.aggregate import (
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
from picr_ma.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]
# Inputs produced by the evaluation stages, and the make target that produces each.
REQUIRED = {
    "main/{env}/crossplay.csv": "crossplay-main",
    "main/{env}/order_shift.csv": "order-shift",
    "lambda/{env}/crossplay.csv": "lambda",
    "lambda/selection.json": "select",
    "reference_returns.json": "reference",
    "positive_control.json": "positive-control",
    "env_speed.json": "env-speed",
    "overhead.json": "overhead",
}


def check_inputs(results_dir: Path, envs: list[str], long_envs: list[str]) -> None:
    required = {**{pattern.format(env=env): t for pattern, t in REQUIRED.items() for env in envs}}
    required.update({f"long/{env}/{name}": "long" for env in long_envs for name in ("crossplay.csv", "order_shift.csv")})
    missing = sorted(f"{path}  (make {target})" for path, target in required.items() if not (results_dir / path).exists())
    if missing:
        raise SystemExit("Missing inputs in " + str(results_dir) + ":\n  " + "\n  ".join(missing))


def input_provenance(results_dir: Path) -> dict[str, str | None]:
    """The git commit recorded by each evaluation output."""
    out = {}
    for path in sorted(results_dir.glob("**/*.json")):
        if path.name in ("analysis.json", "summary.json", "config.json"):
            continue
        record = json.loads(path.read_text())
        if isinstance(record, dict) and "provenance" in record:
            out[str(path.relative_to(results_dir))] = record["provenance"]["git_commit"]
    return out


def compute_record(summaries: pd.DataFrame) -> dict:
    """Training wall-clock per run (single-threaded processes, several in parallel), by environment and method."""
    minutes = summaries.assign(minutes=summaries.train_seconds / 60)
    by_group = minutes.groupby(["env", "variant"]).minutes.agg(["median", "min", "max", "count"]).round(1)
    return {
        "per_run_minutes": {
            env: {variant: row.to_dict() for (e, variant), row in by_group.iterrows() if e == env} for env in set(summaries.env)
        },
        "total_hours": round(float(minutes.minutes.sum() / 60), 1),
    }


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


def check_selected_lambda(summaries: pd.DataFrame, selection_path: Path) -> dict[str, float]:
    """The PICR runs must have been trained with the weight the selection rule picked."""
    selected = {env: v["selected_lambda"] for env, v in json.loads(selection_path.read_text()).items()}
    picr = summaries[summaries.variant == "picr"]
    used = {env: sorted(set(picr[picr.env == env].picr_lambda)) for env in set(picr.env)}
    if any(used[env] != [selected[env]] for env in used):
        raise SystemExit(f"Main-suite PICR runs were trained with {used}, but {selection_path} selects {selected}")
    return selected


def per_run_table(summaries: pd.DataFrame, xp: pd.DataFrame, env: str) -> pd.DataFrame:
    """Per run: permutation sensitivity and competence-adjusted compatibility with its partners."""
    runs = summaries[summaries.env == env][["variant", "seed", "perm_js"]].rename(columns={"variant": "method"})
    return runs.merge(run_compatibility(xp), on=["method", "seed"])


def within_method_spearman(per_run: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation between a run's permutation sensitivity and its compatibility, within each method."""
    out = {}
    for method, runs in per_run.groupby("method"):
        if runs.perm_js.std() > 0:
            out[method] = float(runs[["perm_js", "compatibility"]].corr("spearman").iloc[0, 1])
    return out


def late_training_slope(curves: pd.DataFrame, window: float = 1e6) -> dict[str, dict]:
    """Per method, the least-squares slope of training return over each run's last ``window`` steps (points per 1M steps)."""
    out = {}
    for variant, runs in curves.groupby("variant"):
        slopes = []
        for _, run in runs.groupby("seed"):
            last = run[run.env_steps > run.env_steps.max() - window]
            slopes.append(np.polyfit(last.env_steps / 1e6, last.episode_return, 1)[0])
        slopes = np.array(slopes)
        out[variant] = {
            "mean": float(slopes.mean()),
            "sd": float(slopes.std(ddof=1)) if len(slopes) > 1 else 0.0,
            "n_positive": int((slopes > 0).sum()),
            "n": len(slopes),
        }
    return out


def lambda_table(suite_dir: Path) -> pd.DataFrame:
    summaries = load_summaries(suite_dir)
    rows = []
    for env in sorted(set(summaries.env)):
        xp = load_crossplay(suite_dir, env)
        for variant in sorted(set(summaries[summaries.env == env].variant)):
            lam = float(variant.split("_lam")[1]) if "_lam" in variant else 0.0  # IPPO is the lambda = 0 reference
            block = crossplay_block(xp, variant, variant)
            runs = summaries[(summaries.env == env) & (summaries.variant == variant)]
            sp, cross, js = bootstrap_mean(np.diag(block)), bootstrap_pairs(block, True), bootstrap_mean(runs.perm_js)
            rows.append(
                {
                    "env": env,
                    "method": variant.split("_lam")[0],
                    "lam": lam,
                    "n_seeds": len(runs),
                    "gap": sp.mean - cross.mean,
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
    parser.add_argument(
        "--png-dir", type=Path, default=ROOT / "docs" / "figures", help="PNG copies of the figures, for the README"
    )
    parser.add_argument("--n-perm", type=int, default=10_000, help="permutations per permutation test")
    args = parser.parse_args()
    figures.PNG_DIR = args.png_dir
    fig_dir, tab_dir = args.paper_dir / "figures", args.paper_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    main_dir, long_dir = args.results_dir / "main", args.results_dir / "long"
    suites = load_suites(args.config)["suites"]
    for suite in ("main", "lambda", "long"):
        if suite in suites:
            check_complete(suite, args.results_dir, args.config)
    summaries = load_summaries(main_dir)
    envs = [e for e in ("simple_spread_v3", "simple_adversary_v3") if e in set(summaries.env)]
    long_envs = [e for e in envs if e in suites.get("long", {}).get("envs", [])]
    check_inputs(args.results_dir, envs, long_envs)
    estimates, comparisons, curves, per_run, matrices, crossplays = {}, {}, {}, {}, {}, {}
    for env in envs:
        xp = load_crossplay(main_dir, env)
        crossplays[env] = xp
        shift = load_order_shift(main_dir, env)
        estimates[env] = method_estimates(summaries, xp, env, shift=shift)
        comparisons[env] = compare_to(estimates[env], summaries, xp, env, n_perm=args.n_perm, shift=shift)
        curves[env] = load_curves(main_dir, env)
        per_run[env] = per_run_table(summaries, xp, env)
        matrices[env] = method_matrix(xp)

    figures.learning_curves(curves, fig_dir / "learning_curves.pdf")
    figures.sp_vs_xp(estimates, fig_dir / "self_play_vs_cross_play.pdf")
    figures.sensitivity_vs_crossplay(per_run, fig_dir / "sensitivity_vs_cross_play.pdf")
    figures.additive_fit(crossplays, fig_dir / "additive_fit.pdf")
    if any("shift_drop" in e for est in estimates.values() for e in est.values()):
        figures.order_shift(estimates, fig_dir / "order_shift.pdf")
    (tab_dir / "main_results.tex").write_text(tables.main_table(estimates))
    (tab_dir / "comparison.tex").write_text(tables.comparison_table(comparisons))
    (tab_dir / "order_measures.tex").write_text(tables.order_table(estimates))

    report = {
        "seeds_per_method": {env: summaries[summaries.env == env].groupby("variant").seed.nunique().to_dict() for env in envs},
        "estimates": {env: {m: {k: v.as_dict() for k, v in est.items()} for m, est in estimates[env].items()} for env in envs},
        "vs_ippo": comparisons,
        "method_crossplay_matrix_ego_by_partner": {env: matrices[env].round(3).to_dict(orient="index") for env in envs},
        "within_method_spearman_perm_js_vs_compatibility": {env: within_method_spearman(per_run[env]) for env in envs},
        "additive_decomposition": {env: additive_decomposition(crossplays[env]) for env in envs},
        "late_training_slope": {env: late_training_slope(curves[env]) for env in envs},
    }
    report["selected_lambda"] = check_selected_lambda(summaries, args.results_dir / "lambda" / "selection.json")
    for name in ("reference_returns", "overhead", "positive_control", "env_speed"):
        report[name] = json.loads((args.results_dir / f"{name}.json").read_text())
    (tab_dir / "overhead.tex").write_text(tables.overhead_table(report["overhead"]))
    lambda_dir = args.results_dir / "lambda"
    sweep = lambda_table(lambda_dir)
    figures.lambda_sweep(sweep, fig_dir / "lambda_sweep.pdf")
    report["lambda_sweep"] = sweep.round(5).to_dict(orient="records")
    report["compute"] = {"main": compute_record(summaries), "lambda": compute_record(load_summaries(lambda_dir))}
    long = None
    if long_envs:
        long_summaries = load_summaries(long_dir)
        long = ({}, {})
        for env in long_envs:
            xp, shift = load_crossplay(long_dir, env), load_order_shift(long_dir, env)
            long[0][env] = method_estimates(long_summaries, xp, env, shift=shift)
            long[1][env] = compare_to(long[0][env], long_summaries, xp, env, n_perm=args.n_perm, shift=shift)
        (tab_dir / "long_budget.tex").write_text(tables.long_table(*long))
        report["long"] = {
            "estimates": {
                env: {m: {k: v.as_dict() for k, v in est.items()} for m, est in long[0][env].items()} for env in long_envs
            },
            "vs_ippo": long[1],
        }
        report["compute"]["long"] = compute_record(long_summaries)
        report["long"]["late_training_slope"] = {env: late_training_slope(load_curves(long_dir, env)) for env in long_envs}
    report["provenance"] = provenance()
    report["input_provenance"] = input_provenance(args.results_dir)

    (tab_dir / "numbers.tex").write_text(tables.numbers_tex(estimates, comparisons, report, long=long))
    (args.results_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    (args.results_dir / "main_results.md").write_text(tables.markdown_table(estimates))
    print(tables.markdown_table(estimates))


if __name__ == "__main__":
    main()
