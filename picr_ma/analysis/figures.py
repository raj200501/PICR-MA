"""Paper figures (matplotlib, vector PDF).

Colors are a fixed categorical order validated for color-vision deficiency;
every series also has a distinct marker or line style and a legend entry, so
identity never relies on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from picr_ma.analysis.aggregate import ENV_LABELS, METHOD_LABELS, METHOD_ORDER  # noqa: E402

COLORS = {"ippo": "#2a78d6", "picr": "#eb6834", "deepsets": "#1baf7a", "order_aug": "#eda100"}
MARKERS = {"ippo": "o", "picr": "s", "deepsets": "^", "order_aug": "D"}
LINESTYLES = {"ippo": "-", "picr": "-", "deepsets": "--", "order_aug": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _binned(curves: pd.DataFrame, column: str, n_bins: int = 40) -> pd.DataFrame:
    """Average each seed's curve within equal-width step bins, then pool seeds."""
    edges = np.linspace(0, curves.env_steps.max(), n_bins + 1)
    curves = curves.assign(bin=np.clip(np.digitize(curves.env_steps, edges) - 1, 0, n_bins - 1))
    per_seed = curves.groupby(["variant", "seed", "bin"])[column].mean().reset_index()
    stats = per_seed.groupby(["variant", "bin"])[column].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["steps"] = 0.5 * (edges[stats.bin] + edges[stats.bin + 1])
    return stats


def learning_curves(curves_by_env: dict[str, pd.DataFrame], out: Path) -> None:
    """Training return and permutation sensitivity vs environment steps."""
    envs = list(curves_by_env)
    fig, axes = plt.subplots(2, len(envs), figsize=(3.3 * len(envs), 4.2), squeeze=False)
    for col, env in enumerate(envs):
        for row, (column, label) in enumerate([("episode_return", "Training return"), ("perm_js", "Permutation JS")]):
            ax = axes[row, col]
            stats = _binned(curves_by_env[env], column)
            for method in METHOD_ORDER:
                s = stats[stats.variant == method]
                if s.empty or (column == "perm_js" and method == "deepsets"):
                    continue
                ax.plot(
                    s.steps / 1e3, s["mean"], color=COLORS[method], ls=LINESTYLES[method], lw=1.5, label=METHOD_LABELS[method]
                )
                ax.fill_between(s.steps / 1e3, s["mean"] - s.se, s["mean"] + s.se, color=COLORS[method], alpha=0.18, lw=0)
            if column == "perm_js":
                ax.set_yscale("log")
            ax.set_ylabel(label)
            if row == 0:
                title = ENV_LABELS.get(env, env)
                ax.set_title(title + (" (vs. own co-trained adversary)" if env == "simple_adversary_v3" else ""))
            if row == 1:
                ax.set_xlabel("Environment steps (thousands)")
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[1, 0].text(0.02, 0.03, "Deep Sets: 0 by construction", transform=axes[1, 0].transAxes, color=MUTED, fontsize=6.5)
    fig.align_ylabels(axes[:, 0])
    fig.savefig(out)
    plt.close(fig)


def sp_vs_xp(estimates_by_env: dict, out: Path) -> None:
    """Self-play and cross-play return per method with 95% CIs."""
    envs = list(estimates_by_env)
    fig, axes = plt.subplots(1, len(envs), figsize=(3.3 * len(envs), 2.2), squeeze=False)
    for ax, env in zip(axes[0], envs):
        est = estimates_by_env[env]
        methods = [m for m in METHOD_ORDER if m in est]
        for i, method in enumerate(methods):
            for offset, key, filled in [(-0.12, "sp", False), (0.12, "xp", True)]:
                e = est[method][key]
                ax.errorbar(
                    i + offset,
                    e.mean,
                    yerr=[[e.mean - e.lo], [e.hi - e.mean]],
                    fmt=MARKERS[method],
                    color=COLORS[method],
                    mfc=COLORS[method] if filled else "white",
                    ms=6,
                    capsize=0,
                    lw=1.2,
                )
            ax.plot([i - 0.12, i + 0.12], [est[method]["sp"].mean, est[method]["xp"].mean], color=MUTED, lw=0.6, zorder=0)
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        ax.set_title(ENV_LABELS.get(env, env))
        ax.set_ylabel("Team return")
        ax.grid(axis="x", visible=False)
    handles = [
        plt.Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", label="Self-play"),
        plt.Line2D([], [], marker="o", color=MUTED, ls="", label="Cross-play"),
    ]
    axes[0, -1].legend(handles=handles, frameon=False, loc="best")
    fig.savefig(out)
    plt.close(fig)


def sensitivity_vs_crossplay(per_run_by_env: dict[str, pd.DataFrame], out: Path) -> None:
    """Each point is one run: its permutation sensitivity vs its mean cross-play return."""
    envs = list(per_run_by_env)
    fig, axes = plt.subplots(1, len(envs), figsize=(3.3 * len(envs), 2.4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        runs = per_run_by_env[env]
        for method in METHOD_ORDER:
            r = runs[runs.method == method]
            if r.empty:
                continue
            x = np.maximum(r.perm_js, 1e-6)
            ax.scatter(
                x,
                r.xp,
                marker=MARKERS[method],
                s=22,
                color=COLORS[method],
                edgecolor="white",
                lw=0.6,
                label=METHOD_LABELS[method],
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Permutation JS (log; Deep Sets clipped to 1e-6)")
        ax.set_ylabel("Cross-play return")
        ax.set_title(ENV_LABELS.get(env, env))
    axes[0, 0].legend(frameon=False, loc="best")
    fig.savefig(out)
    plt.close(fig)


def lambda_sweep(sweep: pd.DataFrame, out: Path) -> None:
    """Permutation sensitivity and returns as a function of the PICR weight."""
    envs = list(dict.fromkeys(sweep.env))
    fig, axes = plt.subplots(2, len(envs), figsize=(3.3 * len(envs), 3.8), squeeze=False, sharex=True)
    for col, env in enumerate(envs):
        s = sweep[sweep.env == env].sort_values("lam")
        ax = axes[0, col]
        ax.errorbar(
            s.lam,
            s.perm_js,
            yerr=[s.perm_js - s.perm_js_lo, s.perm_js_hi - s.perm_js],
            color=COLORS["picr"],
            marker="s",
            ms=5,
            lw=1.2,
            capsize=0,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Permutation JS")
        ax.set_title(ENV_LABELS.get(env, env))
        ax = axes[1, col]
        for key, filled, label in [("sp", False, "Self-play"), ("xp", True, "Cross-play")]:
            ax.errorbar(
                s.lam,
                s[key],
                yerr=[s[key] - s[f"{key}_lo"], s[f"{key}_hi"] - s[key]],
                color=COLORS["picr"],
                mfc=COLORS["picr"] if filled else "white",
                marker="s",
                ms=5,
                lw=1.2,
                ls="-" if filled else "--",
                capsize=0,
                label=label,
            )
        ax.set_ylabel("Team return")
        ax.set_xlabel(r"PICR weight $\lambda$")
        ax.set_xscale("symlog", linthresh=0.01)
    axes[1, 0].legend(frameon=False, loc="best")
    fig.align_ylabels(axes[:, 0])
    fig.savefig(out)
    plt.close(fig)
