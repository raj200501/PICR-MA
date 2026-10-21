"""LaTeX and Markdown tables generated from analysis results."""

from __future__ import annotations

from picr_ma.analysis.aggregate import ENV_LABELS, METHOD_LABELS, METHOD_ORDER


def _ci(e, fmt: str = "{:.2f}", scale: float = 1.0) -> str:
    return f"{fmt.format(scale * e.mean)} \\scriptsize{{[{fmt.format(scale * e.lo)}, {fmt.format(scale * e.hi)}]}}"


def main_table(estimates_by_env: dict) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Env. & Method & Self-play $\uparrow$ & Cross-play $\uparrow$ & SP$-$XP gap $\downarrow$ & Perm.\ JS ($\times 10^{-3}$) $\downarrow$ & Flip rate $\downarrow$ \\",
        r"\midrule",
    ]
    for k, (env, est) in enumerate(estimates_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            lines.append(
                f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & {_ci(e['sp'])} & {_ci(e['xp'])} & {_ci(e['gap'])} & "
                f"{_ci(e['perm_js'], '{:.1f}', 1e3)} & {_ci(e['flip'], '{:.2f}')} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _p(p: float) -> str:
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def comparison_table(comparisons_by_env: dict) -> str:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Env. & Method & $\Delta$ Cross-play & $\Delta$ SP$-$XP gap & $\Delta$ Perm.\ JS ($\times 10^{-3}$) \\",
        r"\midrule",
    ]
    for k, (env, comp) in enumerate(comparisons_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in comp]:
            cells = []
            for key, scale, fmt in [("xp", 1, "{:+.2f}"), ("gap", 1, "{:+.2f}"), ("perm_js", 1e3, "{:+.1f}")]:
                d = comp[method][key]
                cells.append(
                    f"{fmt.format(scale * d['diff'])} \\scriptsize{{[{fmt.format(scale * d['ci_lo'])}, {fmt.format(scale * d['ci_hi'])}], $p$={_p(d['p'])}}}"
                )
            lines.append(f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown_table(estimates_by_env: dict) -> str:
    rows = [
        "| Env | Method | Self-play ↑ | Cross-play ↑ | SP−XP gap ↓ | Perm. JS (×10⁻³) ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for env, est in estimates_by_env.items():
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            fmt = lambda x, s=1.0, f="{:.2f}": f"{f.format(s * x.mean)} [{f.format(s * x.lo)}, {f.format(s * x.hi)}]"  # noqa: E731
            rows.append(
                f"| {ENV_LABELS.get(env, env)} | {METHOD_LABELS[method]} | {fmt(e['sp'])} | {fmt(e['xp'])} | {fmt(e['gap'])} | {fmt(e['perm_js'], 1e3, '{:.1f}')} |"
            )
    return "\n".join(rows) + "\n"
