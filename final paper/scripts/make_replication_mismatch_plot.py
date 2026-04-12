#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.statistics import bootstrap_mean_ci
from src.eval.partner_pool import load_run_summaries


COHORTS = [
    ("hardmain50k", "Original", "#4C78A8", "o"),
    ("paired50k_ext", "Replication", "#F58518", "^"),
]

ENVS = [
    ("simple_spread_v3", "Simple Spread"),
    ("simple_adversary_v3", "Simple Adversary"),
]


def paired_df(results_dir: Path, run_tag: str, env: str) -> pd.DataFrame:
    rows = load_run_summaries(results_dir, env=env, run_tag=run_tag, methods=["ippo", "picr_ma"])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ippo = df[df["method"] == "ippo"].set_index("seed")
    picr = df[df["method"] == "picr_ma"].set_index("seed")
    shared = sorted(set(ippo.index) & set(picr.index))
    out = []
    for seed in shared:
        out.append(
            {
                "seed": int(seed),
                "delta_crossplay_return": float(picr.loc[seed, "crossplay_return"] - ippo.loc[seed, "crossplay_return"]),
                "delta_permutation_js": float(picr.loc[seed, "permutation_js"] - ippo.loc[seed, "permutation_js"]),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    results_dir = ROOT / "results"
    out_path = ROOT / "latex/figures/invariance_coordination_mismatch.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (env, title) in zip(axes, ENVS):
        pooled = []
        ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        for run_tag, cohort_label, color, marker in COHORTS:
            df = paired_df(results_dir, run_tag, env)
            if df.empty:
                continue
            pooled.append(df)
            ax.scatter(
                df["delta_permutation_js"],
                df["delta_crossplay_return"],
                color=color,
                marker=marker,
                s=52,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.7,
                label=cohort_label,
                zorder=3,
            )
            for _, row in df.iterrows():
                ax.annotate(
                    f"s{int(row['seed'])}",
                    (row["delta_permutation_js"], row["delta_crossplay_return"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="#333333",
                )

        if pooled:
            pooled_df = pd.concat(pooled, ignore_index=True)
            mean_js, js_lo, js_hi = bootstrap_mean_ci(pooled_df["delta_permutation_js"])
            mean_cross, cross_lo, cross_hi = bootstrap_mean_ci(pooled_df["delta_crossplay_return"])
            ax.errorbar(
                [mean_js],
                [mean_cross],
                xerr=[[mean_js - js_lo], [js_hi - mean_js]],
                yerr=[[mean_cross - cross_lo], [cross_hi - mean_cross]],
                fmt="D",
                color="#111111",
                markersize=7,
                markerfacecolor="#111111",
                markeredgecolor="white",
                capsize=3,
                label="Pooled mean" if env == ENVS[0][0] else None,
                zorder=4,
            )
            js_lower = int((pooled_df["delta_permutation_js"] < 0.0).sum())
            cross_up = int((pooled_df["delta_crossplay_return"] > 0.0).sum())
            ax.text(
                0.03,
                0.97,
                f"Pooled JS lower: {js_lower}/{len(pooled_df)}\nPooled cross-play up: {cross_up}/{len(pooled_df)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25"},
            )

        ax.set_title(title)
        ax.set_xlabel("$\\Delta$ permutation sensitivity (PICR-MA $-$ IPPO)")
        ax.set_ylabel("$\\Delta$ same-family cross-play (PICR-MA $-$ IPPO)")
        ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("The diagnostic shift replicates more cleanly than the coordination effect", y=1.12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(str(out_path))


if __name__ == "__main__":
    main()
