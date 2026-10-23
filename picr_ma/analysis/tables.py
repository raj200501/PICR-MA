"""LaTeX and Markdown tables generated from analysis results."""

from __future__ import annotations

from picr_ma.analysis.aggregate import ENV_LABELS, METHOD_LABELS, METHOD_ORDER


def _ci(e, fmt: str = "{:.2f}", scale: float = 1.0) -> str:
    return f"{fmt.format(scale * e.mean)} \\scriptsize{{[{fmt.format(scale * e.lo)}, {fmt.format(scale * e.hi)}]}}"


def main_table(estimates_by_env: dict) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Env. & Method & Self-play $\uparrow$ & Cross-play $\uparrow$ & SP$-$XP gap $\downarrow$ & Perm.\ JS ($\times 10^{-3}$) $\downarrow$ & Order-shift drop $\downarrow$ \\",
        r"\midrule",
    ]
    for k, (env, est) in enumerate(estimates_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            shift = _ci(e["shift_drop"]) if "shift_drop" in e else "--"
            lines.append(
                f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & {_ci(e['sp'])} & {_ci(e['xp'])} & {_ci(e['gap'])} & "
                f"{_ci(e['perm_js'], '{:.1f}', 1e3)} & {shift} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _p(p: float) -> str:
    return f"{p:.3f}" if p >= 1e-3 else "$<$0.001"


def comparison_table(comparisons_by_env: dict) -> str:
    columns = [("xp", 1, "{:+.2f}", r"$\Delta$ Cross-play"), ("gap", 1, "{:+.2f}", r"$\Delta$ SP$-$XP gap")]
    columns += [
        ("perm_js", 1e3, "{:+.1f}", r"$\Delta$ Perm.\ JS ($\times 10^{-3}$)"),
        ("shift_drop", 1, "{:+.2f}", r"$\Delta$ Order-shift drop"),
    ]
    first = next(iter(comparisons_by_env.values()))
    columns = [c for c in columns if all(c[0] in m for m in first.values())]
    lines = [
        r"\begin{tabular}{ll" + "c" * len(columns) + "}",
        r"\toprule",
        "Env. & Method & " + " & ".join(c[3] for c in columns) + r" \\",
        r"\midrule",
    ]
    for k, (env, comp) in enumerate(comparisons_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in comp]:
            cells = []
            for key, scale, fmt, _ in columns:
                d = comp[method][key]
                cells.append(
                    f"{fmt.format(scale * d['diff'])} \\scriptsize{{[{fmt.format(scale * d['ci_lo'])}, {fmt.format(scale * d['ci_hi'])}], $p$={_p(d['p'])}}}"
                )
            lines.append(f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown_table(estimates_by_env: dict) -> str:
    rows = [
        "| Env | Method | Self-play ↑ | Cross-play ↑ | SP−XP gap ↓ | Perm. JS (×10⁻³) ↓ | Order-shift drop ↓ |",
        "|---|---|---|---|---|---|---|",
    ]
    fmt = lambda x, s=1.0, f="{:.2f}": f"{f.format(s * x.mean)} [{f.format(s * x.lo)}, {f.format(s * x.hi)}]"  # noqa: E731
    for env, est in estimates_by_env.items():
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            shift = fmt(e["shift_drop"]) if "shift_drop" in e else "–"
            rows.append(
                f"| {ENV_LABELS.get(env, env)} | {METHOD_LABELS[method]} | {fmt(e['sp'])} | {fmt(e['xp'])} | {fmt(e['gap'])} | "
                f"{fmt(e['perm_js'], 1e3, '{:.1f}')} | {shift} |"
            )
    return "\n".join(rows) + "\n"


def overhead_table(overhead: dict) -> str:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Env. & Method & PPO update (s) & Inference forward (ms) & Actor parameters \\",
        r"\midrule",
    ]
    for k, (env, methods) in enumerate(overhead["envs"].items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in methods]:
            r = methods[method]
            lines.append(
                f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & {r['update_seconds']:.2f} ({r['update_vs_ippo']:.2f}$\\times$) & "
                f"{r['forward_ms']:.3f} ({r['forward_vs_ippo']:.2f}$\\times$) & {r['actor_params']} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


ENV_MACRO = {"simple_spread_v3": "Spread", "simple_adversary_v3": "Adversary"}
METHOD_MACRO = {"ippo": "IPPO", "picr": "PICR", "deepsets": "DeepSets", "order_aug": "OrderAug"}
METRIC_MACRO = {
    "sp": ("SP", 1.0, "{:.2f}"),
    "xp": ("XP", 1.0, "{:.2f}"),
    "gap": ("Gap", 1.0, "{:.2f}"),
    "perm_js": ("JS", 1e3, "{:.1f}"),
    "flip": ("Flip", 1.0, "{:.2f}"),
    "shift_drop": ("Shift", 1.0, "{:.2f}"),
    "ablation_drop": ("Ablation", 1.0, "{:.2f}"),
}


def _num(value: float, fmt: str, signed: bool = False) -> str:
    """Math-mode number, e.g. $-1.23$ or (signed) $+0.40$."""
    return "$" + (("{:+" + fmt[2:]) if signed else fmt).format(value) + "$"


def numbers_tex(estimates_by_env: dict, comparisons_by_env: dict, report: dict) -> str:
    """LaTeX macros for every number quoted in the paper text (generated; do not edit)."""
    lines = ["% Generated by scripts/analyze.py from results/. Do not edit by hand."]

    def define(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for env, est in estimates_by_env.items():
        e_name = ENV_MACRO[env]
        for method, metrics in est.items():
            m_name = METHOD_MACRO[method]
            for key, (k_name, scale, fmt) in METRIC_MACRO.items():
                if key not in metrics:
                    continue
                x = metrics[key]
                define(f"{e_name}{m_name}{k_name}", _num(scale * x.mean, fmt))
                define(f"{e_name}{m_name}{k_name}ci", f"[{fmt.format(scale * x.lo)}, {fmt.format(scale * x.hi)}]")
        for method, comp in comparisons_by_env.get(env, {}).items():
            m_name = METHOD_MACRO[method]
            for key, (k_name, scale, fmt) in METRIC_MACRO.items():
                if key not in comp:
                    continue
                d = comp[key]
                define(f"{e_name}{m_name}d{k_name}", _num(scale * d["diff"], fmt, signed=True))
                define(f"{e_name}{m_name}d{k_name}ci", f"[{fmt.format(scale * d['ci_lo'])}, {fmt.format(scale * d['ci_hi'])}]")
                define(f"{e_name}{m_name}d{k_name}p", f"$p{{=}}{d['p']:.3f}$" if d["p"] >= 1e-3 else "$p{<}0.001$")
        for method, rho in report.get("within_method_spearman_perm_js_vs_xp", {}).get(env, {}).items():
            define(f"{e_name}{METHOD_MACRO[method]}Rho", _num(rho, "{:.2f}"))
    for env, lam in report.get("selected_lambda", {}).items():
        define(f"{ENV_MACRO[env]}Lambda", f"{lam:g}")
    ref = report.get("reference_returns", {})
    for key, name in [("random", "RefRandom"), ("noop", "RefNoop"), ("assignment", "RefAssign")]:
        if key in ref:
            define(name, _num(ref[key]["mean"], "{:.2f}"))
    return "\n".join(lines) + "\n"
