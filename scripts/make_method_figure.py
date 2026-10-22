#!/usr/bin/env python
"""Draw the PICR method overview (paper/figures/method_overview.pdf)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#f4f3ef"
SELF, MATE_A, MATE_B = "#c3c2b7", "#2a78d6", "#eb6834"
PICR = "#eb6834"


def obs_vector(ax, x, y, order, label):
    """Draw [self | teammate | teammate] as coloured segments."""
    widths = [0.55, 0.38, 0.38]
    colors = [SELF] + [MATE_A if k == 0 else MATE_B for k in order]
    names = ["self", *[f"$u_{k + 1}$" for k in order]]
    cx = x
    for w, c, n in zip(widths, colors, names):
        ax.add_patch(Rectangle((cx, y), w, 0.32, facecolor=c, edgecolor="white", lw=1.5))
        ax.text(cx + w / 2, y + 0.16, n, ha="center", va="center", fontsize=7.5, color=INK if c == SELF else "white")
        cx += w
    ax.text(x - 0.08, y + 0.16, label, ha="right", va="center", fontsize=8, color=INK)
    return x + sum(widths)


def box(ax, x, y, w, h, text, color=INK, fill=SURFACE):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=fill, edgecolor=color, lw=1)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9, shrinkA=0, shrinkB=0))


def bars(ax, x, y, probs, color):
    for i, p in enumerate(probs):
        ax.add_patch(Rectangle((x + i * 0.13, y), 0.1, 0.85 * p, facecolor=color, edgecolor="none"))
    ax.plot([x - 0.02, x + 0.65], [y, y], color=MUTED, lw=0.6)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 1.95))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-0.05, 3.0)
    ax.axis("off")

    end = obs_vector(ax, 0.85, 1.3, (0, 1), "$o$")
    ax.text(0.85, 1.05, "agent observation", fontsize=7.5, color=MUTED)

    rows = [
        (2.15, (1, 0), r"$\sigma_1 o$", [0.08, 0.62, 0.12, 0.10, 0.08]),
        (0.45, (0, 1), r"$\sigma_2 o$", [0.08, 0.18, 0.50, 0.14, 0.10]),
    ]
    for y, order, label, probs in rows:
        arrow(ax, (end + 0.05, 1.46), (3.05, y + 0.16))
        vend = obs_vector(ax, 3.55, y, order, label)
        arrow(ax, (vend + 0.05, y + 0.16), (5.35, y + 0.16))
        box(ax, 5.4, y - 0.05, 1.25, 0.42, r"shared $\pi_\theta$")
        arrow(ax, (6.7, y + 0.16), (7.05, y + 0.16))
        bars(ax, 7.15, y - 0.03, probs, MUTED)
    ax.text(7.15, 2.72, r"$\pi_\theta(\cdot\mid\sigma_1 o)$", fontsize=7.5, color=MUTED)
    ax.text(7.15, 0.0, r"$\pi_\theta(\cdot\mid\sigma_2 o)$", fontsize=7.5, color=MUTED)
    ax.text(3.55, 2.72, r"$\sigma_1,\sigma_2 \sim \mathrm{Unif}(S_{n-1})$", fontsize=7.5, color=MUTED)

    arrow(ax, (7.95, 2.3), (8.35, 1.62), PICR)
    arrow(ax, (7.95, 0.55), (8.35, 1.3), PICR)
    box(ax, 8.4, 1.12, 1.75, 0.68, "symmetric KL\n" + r"$\mathcal{L}_\mathrm{PICR}$ (train only)", color=PICR, fill="#fdf0ea")
    fig.savefig(ROOT / "paper" / "figures" / "method_overview.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
