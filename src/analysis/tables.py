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


def _pretty_env(env: str) -> str:
    mapping = {
        "simple_spread_v3": "Simple Spread",
        "simple_adversary_v3": "Simple Adversary",
        "matrix_coordination": "Matrix Coordination",
        "matrix_stag_hunt": "Matrix Stag Hunt",
    }
    return mapping.get(env, env.replace("_", " ").title())


def _pretty_method(method: str) -> str:
    mapping = {
        "ippo": "IPPO",
        "picr_ma": "PICR-MA",
        "deepsets": "Deep Sets",
        "otherplay": "Other-Play",
    }
    return mapping.get(method, method)


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
                f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(method))} & {to_pm(train_mean, train_se)} & "
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
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Environment & Ego method & Same-family & Cross-family \\\\",
        "\\midrule",
    ]
    for (env, ego_method), group in crossplay_df.groupby(["env", "ego_method"]):
        same = group[group["partner_method"] == ego_method]["crossplay_return"]
        cross = group[group["partner_method"] != ego_method]["crossplay_return"]
        lines.append(
            f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(ego_method))} & "
            f"{to_pm(*mean_se(same)) if len(same) else '--'} & {to_pm(*mean_se(cross)) if len(cross) else '--'} \\\\"
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
        lines.append(f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(method))} & {to_pm(mean, se)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _runtime_table(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return _empty_table("No runtime results available.")
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Environment & Method & Runtime (s) & Relative to IPPO \\\\",
        "\\midrule",
    ]
    grouped = summary_df.groupby(["env", "method"])["wall_clock_runtime_sec"].mean().reset_index()
    for env in sorted(grouped["env"].unique()):
        env_df = grouped[grouped["env"] == env].copy()
        baseline = float(env_df.loc[env_df["method"] == "ippo", "wall_clock_runtime_sec"].iloc[0]) if any(env_df["method"] == "ippo") else None
        for _, row in env_df.iterrows():
            relative = row["wall_clock_runtime_sec"] / baseline if baseline else float("nan")
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(row['method']))} & "
                f"{row['wall_clock_runtime_sec']:.3f} & {relative:.2f}$\\times$ \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _ablation_table(results_dir: str | Path) -> str:
    rows = []
    for summary_path in sorted(Path(results_dir).glob("run_*/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(row.get("run_tag", "")).startswith("lambda20k") or str(row.get("run_tag", "")) == "drop20k_p010":
            config_path = summary_path.parent / "config_expanded.yaml"
            if config_path.exists():
                config_text = config_path.read_text(encoding="utf-8").splitlines()
                if "picr_lambda" not in row:
                    for line in config_text:
                        if line.startswith("picr_lambda:"):
                            row["picr_lambda"] = float(line.split(":", 1)[1].strip())
                            break
                if "teammate_dropout_p" not in row:
                    for line in config_text:
                        if line.startswith("teammate_dropout_p:"):
                            row["teammate_dropout_p"] = float(line.split(":", 1)[1].strip())
                            break
            rows.append(row)
    ablation_df = pd.DataFrame(rows)
    if ablation_df.empty:
        return _empty_table("No PICR-MA ablation results available.")

    def setting_label(row: pd.Series) -> str:
        if row.get("teammate_dropout_p", 0.0) and float(row.get("teammate_dropout_p", 0.0)) > 0:
            return f"dropout $p={float(row['teammate_dropout_p']):.2f}$"
        return f"$\\lambda={float(row.get('picr_lambda', 0.0)):.2f}$"

    ablation_df["setting"] = ablation_df.apply(setting_label, axis=1)
    setting_order = ["$\\lambda=0.02$", "$\\lambda=0.05$", "$\\lambda=0.10$", "dropout $p=0.10$"]
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Environment & Setting & Cross-play & JS $\\downarrow$ & Runtime (s) \\\\",
        "\\midrule",
    ]
    for env in ["simple_spread_v3", "simple_adversary_v3"]:
        env_df = ablation_df[ablation_df["env"] == env]
        if env_df.empty:
            continue
        for setting in setting_order:
            group = env_df[env_df["setting"] == setting]
            if group.empty:
                continue
            cross_mean, cross_se = mean_se(group["crossplay_return"].dropna())
            js_mean, js_se = mean_se(group["permutation_js"].dropna())
            runtime_mean, _ = mean_se(group["wall_clock_runtime_sec"].dropna())
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {setting} & {to_pm(cross_mean, cross_se)} & "
                f"{to_pm(js_mean, js_se)} & {runtime_mean:.3f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
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
        "ablation_results.tex": _ablation_table(results_dir),
    }
    for name, content in tables.items():
        _write(outdir / name, content)
    return {name: str(outdir / name) for name in tables}
