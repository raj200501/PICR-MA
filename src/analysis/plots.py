from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.analysis.aggregate import build_aggregate_outputs
from src.analysis.statistics import mean_se


METHOD_ORDER = ["ippo", "picr_ma", "deepsets", "otherplay"]
METHOD_LABELS = {
    "ippo": "IPPO",
    "picr_ma": "PICR-MA",
    "deepsets": "Deep Sets",
    "otherplay": "Other-Play",
}
METHOD_COLORS = {
    "ippo": "#4C78A8",
    "picr_ma": "#E45756",
    "deepsets": "#54A24B",
    "otherplay": "#B279A2",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def method_overview_figure(outdir: str | Path) -> str:
    _configure_style()
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.35, 0.18, 0.28, "Self + teammate\nfeature blocks"),
        (0.29, 0.58, 0.18, 0.2, "Permutation $\\pi_1$"),
        (0.29, 0.14, 0.18, 0.2, "Permutation $\\pi_2$"),
        (0.55, 0.58, 0.18, 0.2, "Shared policy"),
        (0.55, 0.14, 0.18, 0.2, "Shared policy"),
        (0.79, 0.36, 0.16, 0.25, "Symmetric KL\nconsistency"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F8F6F1", edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")

    arrows = [
        ((0.23, 0.49), (0.29, 0.67)),
        ((0.23, 0.49), (0.29, 0.24)),
        ((0.47, 0.67), (0.55, 0.67)),
        ((0.47, 0.24), (0.55, 0.24)),
        ((0.73, 0.67), (0.79, 0.49)),
        ((0.73, 0.24), (0.79, 0.49)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#333333"))
    ax.set_title("PICR-MA: two independently permuted teammate-order views are matched by a symmetric KL penalty")
    path = Path(outdir) / "method_overview.pdf"
    _save(fig, path)
    return str(path)


def learning_curves(train_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    mpe_envs = [env for env in ["simple_spread_v3", "simple_adversary_v3"] if env in set(train_df.get("env", []))]
    if not mpe_envs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No MPE learning curves available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "learning_curves_mpe.pdf"
        _save(fig, path)
        return str(path)

    fig, axes = plt.subplots(1, len(mpe_envs), figsize=(6.5 * len(mpe_envs), 3.6), sharey=False)
    if len(mpe_envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, mpe_envs):
        env_df = train_df[train_df["env"] == env]
        for method in METHOD_ORDER:
            method_df = env_df[env_df["method"] == method]
            if method_df.empty:
                continue
            grouped = method_df.groupby("env_step")["episodic_return"]
            mean = grouped.mean()
            se = grouped.std(ddof=0).fillna(0.0) / np.sqrt(grouped.count().clip(lower=1))
            ax.plot(mean.index, mean.values, label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=2)
            ax.fill_between(mean.index, mean.values - se.values, mean.values + se.values, color=METHOD_COLORS[method], alpha=0.15)
        ax.set_title(env.replace("_v3", "").replace("_", " "))
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episodic return")
        ax.grid(alpha=0.2, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("Training curves on MPE", y=1.05)
    path = Path(outdir) / "learning_curves_mpe.pdf"
    _save(fig, path)
    return str(path)


def crossplay_heatmaps(crossplay_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    envs = sorted(crossplay_df["env"].unique()) if not crossplay_df.empty else []
    if not envs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cross-play results available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "zero_shot_crossplay_results.pdf"
        _save(fig, path)
        return str(path)

    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.8))
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        env_df = crossplay_df[crossplay_df["env"] == env]
        pivot = env_df.pivot_table(index="ego_method", columns="partner_method", values="crossplay_return", aggfunc="mean")
        pivot = pivot.reindex(index=[m for m in METHOD_ORDER if m in pivot.index], columns=[m for m in METHOD_ORDER if m in pivot.columns])
        image = ax.imshow(pivot.values, cmap="YlGnBu")
        ax.set_xticks(range(len(pivot.columns)), [METHOD_LABELS[col] for col in pivot.columns], rotation=40, ha="right")
        ax.set_yticks(range(len(pivot.index)), [METHOD_LABELS[row] for row in pivot.index])
        ax.set_title(env.replace("_v3", "").replace("_", " "))
        for row_index in range(len(pivot.index)):
            for col_index in range(len(pivot.columns)):
                value = pivot.iloc[row_index, col_index]
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Zero-shot cross-play heatmaps", y=1.02)
    path = Path(outdir) / "zero_shot_crossplay_results.pdf"
    _save(fig, path)
    return str(path)


def zsc_summary(summary_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    envs = [env for env in ["simple_spread_v3", "simple_adversary_v3"] if env in set(summary_df.get("env", []))]
    if not envs or summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No zero-shot coordination summary available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "zero_shot_crossplay_summary.pdf"
        _save(fig, path)
        return str(path)

    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.8), sharey=False)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        env_df = summary_df[summary_df["env"] == env]
        methods = [method for method in METHOD_ORDER if method in set(env_df["method"])]
        means = []
        ses = []
        for method in methods:
            method_df = env_df[env_df["method"] == method]["crossplay_return"].dropna()
            mean, se = mean_se(method_df)
            means.append(mean)
            ses.append(se)
        x = np.arange(len(methods))
        ax.errorbar(x, means, yerr=ses, fmt="o", color="#222222", ecolor="#222222", elinewidth=1.1, capsize=3)
        for idx, method in enumerate(methods):
            ax.scatter(x[idx], means[idx], color=METHOD_COLORS[method], s=70, zorder=3)
        ax.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=20)
        ax.set_ylabel("Held-out partner cross-play return")
        ax.set_title(env.replace("_v3", "").replace("_", " "))
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.suptitle("Held-out partner zero-shot coordination summary", y=1.02)
    path = Path(outdir) / "zero_shot_crossplay_summary.pdf"
    _save(fig, path)
    return str(path)


def permutation_plot(sensitivity_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    envs = sorted(sensitivity_df["env"].unique()) if not sensitivity_df.empty else []
    if not envs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No permutation sensitivity results available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "permutation_sensitivity_results.pdf"
        _save(fig, path)
        return str(path)

    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.8), sharey=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        env_df = sensitivity_df[sensitivity_df["env"] == env]
        data = []
        labels = []
        colors = []
        for method in METHOD_ORDER:
            method_df = env_df[env_df["method"] == method]
            if method_df.empty:
                continue
            data.append(method_df["permutation_js"].to_numpy())
            labels.append(METHOD_LABELS[method])
            colors.append(METHOD_COLORS[method])
        if data:
            box = ax.boxplot(data, patch_artist=True, tick_labels=labels)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
        ax.set_title(env.replace("_v3", "").replace("_", " "))
        ax.set_ylabel("JS divergence")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Permutation sensitivity under teammate-order swaps", y=1.02)
    path = Path(outdir) / "permutation_sensitivity_results.pdf"
    _save(fig, path)
    return str(path)


def invariance_coordination_mismatch(summary_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    if summary_df.empty or "crossplay_return" not in summary_df or "permutation_js" not in summary_df:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No invariance/coordination data available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "invariance_coordination_mismatch.pdf"
        _save(fig, path)
        return str(path)

    envs = [env for env in ["simple_spread_v3", "simple_adversary_v3"] if env in set(summary_df["env"])]
    plot_df = summary_df[summary_df["env"].isin(envs)].dropna(subset=["crossplay_return", "permutation_js"])
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No invariance/coordination data available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "invariance_coordination_mismatch.pdf"
        _save(fig, path)
        return str(path)

    fig, axes = plt.subplots(1, len(envs), figsize=(5.0 * len(envs), 3.8), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    legend_handles = []
    for ax, env in zip(axes, envs):
        env_df = plot_df[plot_df["env"] == env]
        for method in METHOD_ORDER:
            method_df = env_df[env_df["method"] == method]
            if method_df.empty:
                continue
            handle = ax.scatter(
                method_df["permutation_js"],
                method_df["crossplay_return"],
                color=METHOD_COLORS[method],
                alpha=0.55,
                s=28,
                linewidth=0.5,
                edgecolors="white",
                label=METHOD_LABELS[method],
            )
            ax.scatter(
                method_df["permutation_js"].mean(),
                method_df["crossplay_return"].mean(),
                color=METHOD_COLORS[method],
                edgecolors="black",
                linewidth=0.8,
                s=90,
                marker="D",
                zorder=4,
            )
            if env == envs[0]:
                legend_handles.append(handle)
        corr = np.nan
        if len(env_df) >= 2 and env_df["permutation_js"].std(ddof=0) > 0 and env_df["crossplay_return"].std(ddof=0) > 0:
            corr = float(np.corrcoef(env_df["permutation_js"], env_df["crossplay_return"])[0, 1])
        corr_text = f"r={corr:.2f}" if not np.isnan(corr) else "r=n/a"
        ax.set_title(f"{env.replace('_v3', '').replace('_', ' ')} ({corr_text})")
        ax.set_xlabel("Permutation sensitivity (JS)")
        ax.set_ylabel("Held-out partner cross-play return")
        ax.grid(alpha=0.2, linewidth=0.5)

    if legend_handles:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=len(legend_handles), frameon=False)
    fig.suptitle("Lower permutation sensitivity does not reliably predict better held-out coordination", y=1.03)
    path = Path(outdir) / "invariance_coordination_mismatch.pdf"
    _save(fig, path)
    return str(path)


def runtime_overhead(summary_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No runtime results available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "runtime_overhead.pdf"
        _save(fig, path)
        return str(path)

    grouped = summary_df.groupby("method")["wall_clock_runtime_sec"].mean()
    methods = [method for method in METHOD_ORDER if method in grouped.index]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    values = [grouped[method] for method in methods]
    ax.bar(range(len(methods)), values, color=[METHOD_COLORS[method] for method in methods])
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods], rotation=20)
    ax.set_ylabel("Wall-clock seconds / run")
    ax.set_title("Runtime overhead across available runs")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    path = Path(outdir) / "runtime_overhead.pdf"
    _save(fig, path)
    return str(path)


def matrix_summary(summary_df: pd.DataFrame, outdir: str | Path) -> str:
    _configure_style()
    matrix_df = summary_df[summary_df["env"].str.startswith("matrix_")] if not summary_df.empty else pd.DataFrame()
    if matrix_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No matrix-game results available", ha="center", va="center")
        ax.axis("off")
        path = Path(outdir) / "optional_matrix_games_summary.pdf"
        _save(fig, path)
        return str(path)

    envs = sorted(matrix_df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(4.8 * len(envs), 3.6), sharey=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        env_df = matrix_df[matrix_df["env"] == env]
        grouped = env_df.groupby("method")["train_return"].mean()
        methods = [method for method in METHOD_ORDER if method in grouped.index]
        ax.bar(range(len(methods)), [grouped[m] for m in methods], color=[METHOD_COLORS[m] for m in methods])
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=20)
        ax.set_title(env.replace("_", " "))
        ax.set_ylabel("Train return")
    path = Path(outdir) / "optional_matrix_games_summary.pdf"
    _save(fig, path)
    return str(path)


def make_publication_plots(results_dir: str | Path = "results", outdir: str | Path = "latex/figures") -> list[str]:
    aggregates = build_aggregate_outputs(results_dir)
    summary_df = aggregates["summary"]
    train_df = aggregates["train"]
    crossplay_df = aggregates["crossplay"]
    sensitivity_df = aggregates["sensitivity"]
    outdir = Path(outdir)
    paths = [
        method_overview_figure(outdir),
        learning_curves(train_df, outdir),
        zsc_summary(summary_df, outdir),
        crossplay_heatmaps(crossplay_df, outdir),
        permutation_plot(sensitivity_df, outdir),
        invariance_coordination_mismatch(summary_df, outdir),
        runtime_overhead(summary_df, outdir),
        matrix_summary(summary_df, outdir),
    ]
    return paths
