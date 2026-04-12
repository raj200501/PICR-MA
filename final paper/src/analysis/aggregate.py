from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.analysis.statistics import bootstrap_mean_ci, mean_se
from src.utils.io_utils import ensure_dir


def aggregate_results(results_dir: str | Path = "results") -> list[dict]:
    rows = []
    for summary_path in sorted(Path(results_dir).glob("run_*/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)
    tags = {row.get("run_tag") for row in rows}
    if any(str(tag).startswith("hardmain") for tag in tags):
        rows = [row for row in rows if str(row.get("run_tag", "")).startswith("hardmain")]
    elif {"stage1", "stage2"} & tags:
        rows = [row for row in rows if row.get("run_tag") in {"stage1", "stage2"}]
    elif {"stage0", "smoke"} & tags:
        rows = [row for row in rows if row.get("run_tag") in {"stage0", "smoke"}]
    else:
        rows = [row for row in rows if row.get("run_tag") != "debug"]
    return rows


def _read_csv_gz(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        try:
            return pd.read_csv(handle)
        except EmptyDataError:
            return pd.DataFrame()


def _build_paired_picr_vs_ippo(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if summary_df.empty:
        return pd.DataFrame(), {}

    rows = []
    stats = {}
    for env in sorted(summary_df["env"].unique()):
        env_df = summary_df[summary_df["env"] == env]
        ippo_df = env_df[env_df["method"] == "ippo"].set_index("seed")
        picr_df = env_df[env_df["method"] == "picr_ma"].set_index("seed")
        shared_seeds = sorted(set(ippo_df.index) & set(picr_df.index))
        if not shared_seeds:
            continue
        env_rows = []
        for seed in shared_seeds:
            ippo_row = ippo_df.loc[seed]
            picr_row = picr_df.loc[seed]
            row = {
                "env": env,
                "seed": int(seed),
                "ippo_crossplay_return": float(ippo_row.get("crossplay_return", 0.0)),
                "picr_crossplay_return": float(picr_row.get("crossplay_return", 0.0)),
                "delta_crossplay_return": float(picr_row.get("crossplay_return", 0.0) - ippo_row.get("crossplay_return", 0.0)),
                "ippo_permutation_js": float(ippo_row.get("permutation_js", 0.0)),
                "picr_permutation_js": float(picr_row.get("permutation_js", 0.0)),
                "delta_permutation_js": float(picr_row.get("permutation_js", 0.0) - ippo_row.get("permutation_js", 0.0)),
                "ippo_train_return": float(ippo_row.get("train_return", 0.0)),
                "picr_train_return": float(picr_row.get("train_return", 0.0)),
                "delta_train_return": float(picr_row.get("train_return", 0.0) - ippo_row.get("train_return", 0.0)),
                "ippo_collision_mean": float(ippo_row.get("collision_mean", 0.0)),
                "picr_collision_mean": float(picr_row.get("collision_mean", 0.0)),
                "delta_collision_mean": float(picr_row.get("collision_mean", 0.0) - ippo_row.get("collision_mean", 0.0)),
                "ippo_goal_margin_mean": float(ippo_row.get("goal_margin_mean", 0.0)),
                "picr_goal_margin_mean": float(picr_row.get("goal_margin_mean", 0.0)),
                "delta_goal_margin_mean": float(picr_row.get("goal_margin_mean", 0.0) - ippo_row.get("goal_margin_mean", 0.0)),
            }
            rows.append(row)
            env_rows.append(row)

        env_df_rows = pd.DataFrame(env_rows)
        cross_mean, cross_lo, cross_hi = bootstrap_mean_ci(env_df_rows["delta_crossplay_return"])
        js_mean, js_lo, js_hi = bootstrap_mean_ci(env_df_rows["delta_permutation_js"])
        stats[env] = {
            "paired_seed_count": int(len(env_df_rows)),
            "crossplay": {
                "mean_delta": cross_mean,
                "ci_low": cross_lo,
                "ci_high": cross_hi,
                "picr_better_count": int((env_df_rows["delta_crossplay_return"] > 0.0).sum()),
            },
            "permutation_js": {
                "mean_delta": js_mean,
                "ci_low": js_lo,
                "ci_high": js_hi,
                "picr_lower_count": int((env_df_rows["delta_permutation_js"] < 0.0).sum()),
            },
        }
    return pd.DataFrame(rows), stats


def _build_implementation_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    rows = []
    grouped = summary_df.groupby(["env", "method"], sort=True)
    for (env, method), group in grouped:
        runtime_mean = float(group["wall_clock_runtime_sec"].mean()) if "wall_clock_runtime_sec" in group else 0.0
        rows.append(
            {
                "env": env,
                "method": method,
                "cooperative_param_count": int(group["cooperative_param_count"].iloc[0]),
                "adversary_param_count": int(group["adversary_param_count"].iloc[0]) if "adversary_param_count" in group else 0,
                "training_steps": int(group["training_steps"].iloc[0]) if "training_steps" in group else 0,
                "eval_episodes": int(group["eval_episodes"].iloc[0]) if "eval_episodes" in group else 0,
                "picr_lambda": float(group["picr_lambda"].iloc[0]) if "picr_lambda" in group else 0.0,
                "teammate_dropout_p": float(group["teammate_dropout_p"].iloc[0]) if "teammate_dropout_p" in group else 0.0,
                "mean_runtime_sec": runtime_mean,
                "inference_architecture_changed": method == "deepsets",
                "extra_train_forward_passes": 2 if method == "picr_ma" else 0,
                "shared_base_mlp_with_ippo": method in {"ippo", "picr_ma", "otherplay"},
            }
        )
    implementation_df = pd.DataFrame(rows)
    for env in implementation_df["env"].unique():
        env_mask = implementation_df["env"] == env
        env_ippo = implementation_df[env_mask & (implementation_df["method"] == "ippo")]
        baseline = float(env_ippo["mean_runtime_sec"].iloc[0]) if not env_ippo.empty else 0.0
        implementation_df.loc[env_mask, "runtime_relative_to_ippo"] = (
            implementation_df.loc[env_mask, "mean_runtime_sec"] / baseline if baseline else 0.0
        )
    return implementation_df


def build_aggregate_outputs(results_dir: str | Path = "results") -> dict[str, pd.DataFrame]:
    results_root = Path(results_dir)
    out_dir = ensure_dir(results_root / "aggregate")
    summary_rows = aggregate_results(results_root)
    summary_df = pd.DataFrame(summary_rows)

    train_frames = []
    crossplay_frames = []
    sensitivity_frames = []
    for row in summary_rows:
        run_dir = Path(row["run_dir"])
        train_df = _read_csv_gz(run_dir / "train_metrics.csv.gz")
        if not train_df.empty:
            train_df["env"] = row["env"]
            train_df["method"] = row["method"]
            train_df["seed"] = row["seed"]
            train_df["run_id"] = row["run_id"]
            train_frames.append(train_df)
        cross_df = _read_csv_gz(run_dir / "eval_crossplay.csv.gz")
        if not cross_df.empty:
            cross_df["env"] = row["env"]
            crossplay_frames.append(cross_df)
        sens_df = _read_csv_gz(run_dir / "eval_permutation_sensitivity.csv.gz")
        if not sens_df.empty:
            sens_df["env"] = row["env"]
            sens_df["method"] = row["method"]
            sens_df["seed"] = row["seed"]
            sens_df["run_id"] = row["run_id"]
            sensitivity_frames.append(sens_df)

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    crossplay_df = pd.concat(crossplay_frames, ignore_index=True) if crossplay_frames else pd.DataFrame()
    sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True) if sensitivity_frames else pd.DataFrame()

    if not summary_df.empty:
        summary_df.to_csv(out_dir / "training_summary.csv", index=False)
        summary_df[["run_id", "env", "method", "seed", "wall_clock_runtime_sec", "artifact_footprint_mb"]].to_csv(
            out_dir / "resource_summary.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(out_dir / "training_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "resource_summary.csv", index=False)

    crossplay_df.to_csv(out_dir / "crossplay_summary.csv", index=False)
    sensitivity_df.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    paired_df, paired_stats = _build_paired_picr_vs_ippo(summary_df)
    paired_df.to_csv(out_dir / "paired_picr_vs_ippo.csv", index=False)
    (out_dir / "paired_picr_vs_ippo_stats.json").write_text(json.dumps(paired_stats, indent=2), encoding="utf-8")

    implementation_df = _build_implementation_summary(summary_df)
    implementation_df.to_csv(out_dir / "implementation_summary.csv", index=False)

    final_metrics = {}
    if not summary_df.empty:
        for (env, method), group in summary_df.groupby(["env", "method"]):
            train_mean, train_se = mean_se(group["train_return"].dropna())
            perm_mean, perm_se = mean_se(group["permutation_js"].dropna())
            cross_values = group["crossplay_return"].dropna() if "crossplay_return" in group else []
            cross_mean, cross_se = mean_se(cross_values)
            final_metrics.setdefault(env, {})[method] = {
                "train_return_mean": train_mean,
                "train_return_se": train_se,
                "crossplay_return_mean": cross_mean if len(cross_values) else None,
                "crossplay_return_se": cross_se if len(cross_values) else None,
                "permutation_js_mean": perm_mean,
                "permutation_js_se": perm_se,
                "collision_mean": float(group["collision_mean"].dropna().mean()) if "collision_mean" in group else 0.0,
                "goal_margin_mean": float(group["goal_margin_mean"].dropna().mean()) if "goal_margin_mean" in group else 0.0,
                "success_mean": float(group["success_mean"].dropna().mean()) if "success_mean" in group else 0.0,
            }
    (out_dir / "final_metrics_for_paper.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return {
        "summary": summary_df,
        "train": train_df,
        "crossplay": crossplay_df,
        "sensitivity": sensitivity_df,
        "paired_picr_vs_ippo": paired_df,
        "implementation_summary": implementation_df,
    }
