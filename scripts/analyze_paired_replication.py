#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picr_ma.analysis.statistics import bootstrap_mean_ci
from picr_ma.eval.partner_pool import load_run_summaries


COHORTS = [
    ("hardmain50k", "Original 5-seed"),
    ("paired50k_ext", "Replication 5-seed"),
]

ENVS = ["simple_spread_v3", "simple_adversary_v3"]


def paired_rows(results_dir: Path, run_tag: str, env: str) -> list[dict]:
    rows = load_run_summaries(results_dir, env=env, run_tag=run_tag, methods=["ippo", "picr_ma"])
    if not rows:
        return []
    df = pd.DataFrame(rows)
    ippo_df = df[df["method"] == "ippo"].set_index("seed")
    picr_df = df[df["method"] == "picr_ma"].set_index("seed")
    shared = sorted(set(ippo_df.index) & set(picr_df.index))
    out = []
    for seed in shared:
        ippo = ippo_df.loc[seed]
        picr = picr_df.loc[seed]
        out.append(
            {
                "env": env,
                "seed": int(seed),
                "run_tag": run_tag,
                "delta_crossplay_return": float(picr["crossplay_return"] - ippo["crossplay_return"]),
                "delta_permutation_js": float(picr["permutation_js"] - ippo["permutation_js"]),
            }
        )
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "paired_seed_count": 0,
            "crossplay": {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "picr_better_count": 0},
            "permutation_js": {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "picr_lower_count": 0},
        }
    df = pd.DataFrame(rows)
    cross_mean, cross_lo, cross_hi = bootstrap_mean_ci(df["delta_crossplay_return"])
    js_mean, js_lo, js_hi = bootstrap_mean_ci(df["delta_permutation_js"])
    return {
        "paired_seed_count": int(len(df)),
        "crossplay": {
            "mean_delta": cross_mean,
            "ci_low": cross_lo,
            "ci_high": cross_hi,
            "picr_better_count": int((df["delta_crossplay_return"] > 0.0).sum()),
        },
        "permutation_js": {
            "mean_delta": js_mean,
            "ci_low": js_lo,
            "ci_high": js_hi,
            "picr_lower_count": int((df["delta_permutation_js"] < 0.0).sum()),
        },
    }


def build_table(summaries: dict) -> str:
    lines = [
        "\\begin{tabular}{lllcc}",
        "\\toprule",
        "Environment & Cohort & Quantity & Mean delta & 95\\% bootstrap CI \\\\",
        "\\midrule",
    ]
    env_labels = {
        "simple_spread_v3": "Simple Spread",
        "simple_adversary_v3": "Simple Adversary",
    }
    for env in ENVS:
        for cohort_key in ["Original 5-seed", "Replication 5-seed", "Pooled 10-pair"]:
            stats = summaries[env][cohort_key]
            lines.append(
                f"{env_labels[env]} & {cohort_key} & Same-family cross-play & "
                f"{stats['crossplay']['mean_delta']:.3f} & "
                f"[{stats['crossplay']['ci_low']:.3f}, {stats['crossplay']['ci_high']:.3f}] \\\\"
            )
            lines.append(
                f"{env_labels[env]} & {cohort_key} & JS $\\downarrow$ & "
                f"{stats['permutation_js']['mean_delta']:.3f} & "
                f"[{stats['permutation_js']['ci_low']:.3f}, {stats['permutation_js']['ci_high']:.3f}] \\\\"
            )
            lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    results_dir = ROOT / "results"
    all_rows: dict[str, dict[str, list[dict]]] = {env: {} for env in ENVS}
    summaries: dict[str, dict[str, dict]] = {env: {} for env in ENVS}

    for env in ENVS:
        pooled_rows: list[dict] = []
        for run_tag, label in COHORTS:
            rows = paired_rows(results_dir, run_tag, env)
            all_rows[env][label] = rows
            summaries[env][label] = summarize(rows)
            pooled_rows.extend(rows)
        summaries[env]["Pooled 10-pair"] = summarize(pooled_rows)

    out_json = ROOT / "results/aggregate/paired_replication_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    out_tex = ROOT / "latex/tables/paired_replication_results.tex"
    out_tex.write_text(build_table(summaries), encoding="utf-8")

    print(str(out_json))
    print(str(out_tex))
