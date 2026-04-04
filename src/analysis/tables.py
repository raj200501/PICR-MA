from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import build_aggregate_outputs
from src.analysis.statistics import mean_se
from src.utils.latex_utils import latex_escape, to_pm


def _write(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _empty_table(message: str) -> str:
    return f"% {message}\n"


def _main_results_table(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return _empty_table("No main results available.")
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Environment & Method & Train return & Cross-play & JS $\\downarrow$ & Env metric \\\\",
        "\\midrule",
    ]
    for env in sorted(summary_df["env"].unique()):
        env_df = summary_df[summary_df["env"] == env]
        for method in [m for m in ["ippo", "picr_ma", "deepsets", "otherplay"] if m in set(env_df["method"])]:
            group = env_df[env_df["method"] == method]
            train_mean, train_se = mean_se(group["train_return"].dropna())
            cross_values = group["crossplay_return"].dropna()
            cross_mean, cross_se = mean_se(cross_values)
            perm_mean, perm_se = mean_se(group["permutation_js"].dropna())
            if env == "simple_spread_v3":
                metric = f"coll={group['collision_mean'].mean():.3f}"
            elif env == "simple_adversary_v3":
                metric = f"margin={group['goal_margin_mean'].mean():.3f}"
            else:
                metric = f"succ={group['success_mean'].mean():.3f}"
            cross_text = to_pm(cross_mean, cross_se) if len(cross_values) else "--"
            lines.append(
                f"{latex_escape(env)} & {latex_escape(method)} & {to_pm(train_mean, train_se)} & "
                f"{cross_text} & {to_pm(perm_mean, perm_se)} & {metric} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _crossplay_table(crossplay_df: pd.DataFrame) -> str:
    if crossplay_df.empty:
        return _empty_table("No cross-play results available.")
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Environment & Ego & Partner & Cross-play return \\\\",
        "\\midrule",
    ]
    grouped = crossplay_df.groupby(["env", "ego_method", "partner_method"])["crossplay_return"].mean().reset_index()
    for _, row in grouped.iterrows():
        lines.append(
            f"{latex_escape(row['env'])} & {latex_escape(row['ego_method'])} & {latex_escape(row['partner_method'])} & {row['crossplay_return']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _sensitivity_table(sensitivity_df: pd.DataFrame) -> str:
    if sensitivity_df.empty:
        return _empty_table("No permutation sensitivity results available.")
    lines = [
        "\\begin{tabular}{llc}",
        "\\toprule",
        "Environment & Method & JS divergence \\\\",
        "\\midrule",
    ]
    grouped = sensitivity_df.groupby(["env", "method"])["permutation_js"]
    for (env, method), values in grouped:
        mean, se = mean_se(values)
        lines.append(f"{latex_escape(env)} & {latex_escape(method)} & {to_pm(mean, se)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _runtime_table(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return _empty_table("No runtime results available.")
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Environment & Method & Runtime (s) & Artifact MB \\\\",
        "\\midrule",
    ]
    grouped = summary_df.groupby(["env", "method"])[["wall_clock_runtime_sec", "artifact_footprint_mb"]].mean().reset_index()
    for _, row in grouped.iterrows():
        lines.append(
            f"{latex_escape(row['env'])} & {latex_escape(row['method'])} & {row['wall_clock_runtime_sec']:.3f} & {row['artifact_footprint_mb']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_tables(results_dir: str | Path = "results", outdir: str | Path = "latex/tables") -> dict[str, str]:
    aggregates = build_aggregate_outputs(results_dir)
    summary_df = aggregates["summary"]
    crossplay_df = aggregates["crossplay"]
    sensitivity_df = aggregates["sensitivity"]
    outdir = Path(outdir)

    tables = {
        "main_results.tex": _main_results_table(summary_df),
        "crossplay_results.tex": _crossplay_table(crossplay_df),
        "permutation_sensitivity.tex": _sensitivity_table(sensitivity_df),
        "runtime_results.tex": _runtime_table(summary_df),
    }
    for name, content in tables.items():
        _write(outdir / name, content)
    return {name: str(outdir / name) for name in tables}
