from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.aggregate import build_aggregate_outputs
from src.analysis.statistics import bootstrap_mean_ci, mean_se
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


def _maybe_bold(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def _main_results_table(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return _empty_table("No main results available.")
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Environment & Method & Train return & Same-family cross-play & JS $\\downarrow$ & Env metric \\\\",
        "\\midrule",
    ]
    for env in sorted(summary_df["env"].unique()):
        env_df = summary_df[summary_df["env"] == env]
        cross_means = env_df.groupby("method")["crossplay_return"].mean()
        perm_means = env_df.groupby("method")["permutation_js"].mean()
        best_cross = float(cross_means.max()) if not cross_means.empty else None
        best_perm = float(perm_means.min()) if not perm_means.empty else None
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
            if len(cross_values) and best_cross is not None:
                cross_text = _maybe_bold(cross_text, np.isclose(cross_mean, best_cross))
            perm_text = to_pm(perm_mean, perm_se)
            if best_perm is not None:
                perm_text = _maybe_bold(perm_text, np.isclose(perm_mean, best_perm))
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(method))} & {to_pm(train_mean, train_se)} & "
                f"{cross_text} & {perm_text} & {metric} \\\\"
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
    order = ["ippo", "picr_ma", "deepsets", "otherplay"]
    for env in sorted(crossplay_df["env"].unique()):
        env_df = crossplay_df[crossplay_df["env"] == env]
        rows = []
        for ego_method, group in env_df.groupby("ego_method"):
            same = group[group["partner_method"] == ego_method]["crossplay_return"]
            cross = group[group["partner_method"] != ego_method]["crossplay_return"]
            same_mean, _ = mean_se(same)
            cross_mean, _ = mean_se(cross)
            rows.append(
                {
                    "ego_method": ego_method,
                    "same_mean": same_mean if len(same) else None,
                    "cross_mean": cross_mean if len(cross) else None,
                    "same_text": to_pm(*mean_se(same)) if len(same) else "--",
                    "cross_text": to_pm(*mean_se(cross)) if len(cross) else "--",
                }
            )
        same_candidates = [row["same_mean"] for row in rows if row["same_mean"] is not None]
        cross_candidates = [row["cross_mean"] for row in rows if row["cross_mean"] is not None]
        best_same = max(same_candidates) if same_candidates else None
        best_cross = max(cross_candidates) if cross_candidates else None
        for row in sorted(rows, key=lambda item: order.index(item["ego_method"]) if item["ego_method"] in order else len(order)):
            same_text = _maybe_bold(
                row["same_text"],
                best_same is not None and row["same_mean"] is not None and np.isclose(row["same_mean"], best_same),
            )
            cross_text = _maybe_bold(
                row["cross_text"],
                best_cross is not None and row["cross_mean"] is not None and np.isclose(row["cross_mean"], best_cross),
            )
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(row['ego_method']))} & "
                f"{same_text} & {cross_text} \\\\"
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
    order = ["ippo", "picr_ma", "deepsets", "otherplay"]
    for env in sorted(sensitivity_df["env"].unique()):
        env_df = sensitivity_df[sensitivity_df["env"] == env]
        means = {method: mean_se(values)[0] for method, values in env_df.groupby("method")["permutation_js"]}
        best_env = min(means.values()) if means else None
        for method in [m for m in order if m in means]:
            values = env_df[env_df["method"] == method]["permutation_js"]
            mean, se = mean_se(values)
            text = _maybe_bold(to_pm(mean, se), best_env is not None and np.isclose(mean, best_env))
            lines.append(f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(method))} & {text} \\\\")
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
        fastest = float(env_df["wall_clock_runtime_sec"].min()) if not env_df.empty else None
        for _, row in env_df.iterrows():
            relative = row["wall_clock_runtime_sec"] / baseline if baseline else float("nan")
            runtime_text = f"{row['wall_clock_runtime_sec']:.3f}"
            runtime_text = _maybe_bold(runtime_text, fastest is not None and np.isclose(row["wall_clock_runtime_sec"], fastest))
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {latex_escape(_pretty_method(row['method']))} & "
                f"{runtime_text} & {relative:.2f}$\\times$ \\\\"
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
        best_cross = float(env_df.groupby("setting")["crossplay_return"].mean().max())
        best_js = float(env_df.groupby("setting")["permutation_js"].mean().min())
        for setting in setting_order:
            group = env_df[env_df["setting"] == setting]
            if group.empty:
                continue
            cross_mean, cross_se = mean_se(group["crossplay_return"].dropna())
            js_mean, js_se = mean_se(group["permutation_js"].dropna())
            runtime_mean, _ = mean_se(group["wall_clock_runtime_sec"].dropna())
            cross_text = _maybe_bold(to_pm(cross_mean, cross_se), np.isclose(cross_mean, best_cross))
            js_text = _maybe_bold(to_pm(js_mean, js_se), np.isclose(js_mean, best_js))
            lines.append(
                f"{latex_escape(_pretty_env(env))} & {setting} & {cross_text} & "
                f"{js_text} & {runtime_mean:.3f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _paired_delta_table(paired_df: pd.DataFrame) -> str:
    if paired_df.empty:
        return _empty_table("No paired PICR-MA versus IPPO results available.")
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Environment & Quantity & Mean delta & 95\\% bootstrap CI & Direction count \\\\",
        "\\midrule",
    ]
    for env in ["simple_spread_v3", "simple_adversary_v3"]:
        env_df = paired_df[paired_df["env"] == env]
        if env_df.empty:
            continue
        cross_mean, cross_lo, cross_hi = bootstrap_mean_ci(env_df["delta_crossplay_return"])
        js_mean, js_lo, js_hi = bootstrap_mean_ci(env_df["delta_permutation_js"])
        lines.append(
            f"{latex_escape(_pretty_env(env))} & Cross-play & {cross_mean:.3f} & "
            f"[{cross_lo:.3f}, {cross_hi:.3f}] & {int((env_df['delta_crossplay_return'] > 0.0).sum())}/{len(env_df)} higher \\\\"
        )
        lines.append(
            f"{latex_escape(_pretty_env(env))} & JS $\\downarrow$ & {js_mean:.3f} & "
            f"[{js_lo:.3f}, {js_hi:.3f}] & {int((env_df['delta_permutation_js'] < 0.0).sum())}/{len(env_df)} lower \\\\"
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
    paired_df = aggregates["paired_picr_vs_ippo"]
    outdir = Path(outdir)

    tables = {
        "main_results.tex": _main_results_table(summary_df),
        "crossplay_results.tex": _crossplay_table(crossplay_df),
        "permutation_sensitivity.tex": _sensitivity_table(sensitivity_df),
        "runtime_results.tex": _runtime_table(summary_df),
        "ablation_results.tex": _ablation_table(results_dir),
        "paired_delta_results.tex": _paired_delta_table(paired_df),
    }
    for name, content in tables.items():
        _write(outdir / name, content)
    return {name: str(outdir / name) for name in tables}
