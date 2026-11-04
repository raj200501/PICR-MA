"""LaTeX and Markdown tables, and the LaTeX number macros, generated from analysis results.

Every number is typeset in math mode so negative values get a proper minus sign.
"""

from __future__ import annotations

from picr_ma.analysis.aggregate import ENV_LABELS, METHOD_LABELS, METHOD_ORDER

ENV_MACRO = {"simple_spread_v3": "Spread", "simple_adversary_v3": "Adversary"}
METHOD_MACRO = {"ippo": "IPPO", "picr": "PICR", "deepsets": "DeepSets", "order_aug": "OrderAug", "order_rand": "OrderRand"}
# metric key -> (macro suffix, scale, format)
METRIC_MACRO = {
    "sp": ("SP", 1.0, "{:.2f}"),
    "xp": ("XP", 1.0, "{:.2f}"),
    "gap": ("Gap", 1.0, "{:.2f}"),
    "perm_js": ("JS", 1e3, "{:.1f}"),
    "flip": ("Flip", 1.0, "{:.2f}"),
    "slot_contrast": ("Slot", 1.0, "{:.2f}"),
    "shift_drop": ("Shift", 1.0, "{:.2f}"),
    "relabel_drop": ("Relabel", 1.0, "{:.2f}"),
    "resample_drop": ("Resample", 1.0, "{:.2f}"),
}


def _num(value: float, fmt: str = "{:.2f}", signed: bool = False) -> str:
    """Math-mode number, e.g. $-1.23$ or (signed) $+0.40$."""
    return "$" + (("{:+" + fmt[2:]) if signed else fmt).format(value) + "$"


def _interval(lo: float, hi: float, fmt: str = "{:.2f}", signed: bool = False) -> str:
    f = ("{:+" + fmt[2:]) if signed else fmt
    return f"$[{f.format(lo)}, {f.format(hi)}]$"


def _p(p: float) -> str:
    return f"${p:.3f}$" if p >= 1e-3 else "$<0.001$"


def _cell(e, key: str) -> str:
    _, scale, fmt = METRIC_MACRO[key]
    return f"{_num(scale * e.mean, fmt)} \\scriptsize{{{_interval(scale * e.lo, scale * e.hi, fmt)}}}"


def _rows(estimates_by_env: dict, keys: list[str]) -> list[str]:
    lines = []
    for k, (env, est) in enumerate(estimates_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            cells = [_cell(e[key], key) if key in e else "--" for key in keys]
            lines.append(f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    return lines


def main_table(estimates_by_env: dict) -> str:
    keys = ["sp", "xp", "gap", "perm_js", "shift_drop"]
    header = (
        r"Env. & Method & Self-play $\uparrow$ & Cross-play $\uparrow$ & SP$-$XP gap $\downarrow$ & "
        r"Perm.\ JS ($\times 10^{-3}$) & Order-shift drop \\"
    )
    lines = [r"\begin{tabular}{ll" + "c" * len(keys) + "}", r"\toprule", header, r"\midrule"]
    lines += _rows(estimates_by_env, keys) + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def order_table(estimates_by_env: dict) -> str:
    """Appendix: every order-related measure, including the controls."""
    keys = ["flip", "slot_contrast", "shift_drop", "relabel_drop", "resample_drop"]
    header = (
        r"Env. & Method & Greedy-action flip rate & Slot contrast & Genuine order shift & "
        r"Relabeling (null) & Teammates resampled \\"
    )
    lines = [r"\begin{tabular}{ll" + "c" * len(keys) + "}", r"\toprule", header, r"\midrule"]
    lines += _rows(estimates_by_env, keys) + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


COMPARISON_COLUMNS = [
    ("gap", r"$\Delta$ SP$-$XP gap (primary)"),
    ("xp", r"$\Delta$ Cross-play"),
    ("perm_js", r"$\Delta$ Perm.\ JS ($\times 10^{-3}$)"),
    ("shift_drop", r"$\Delta$ Order-shift drop"),
]


def comparison_table(comparisons_by_env: dict) -> str:
    first = next(iter(comparisons_by_env.values()))
    columns = [(k, h) for k, h in COMPARISON_COLUMNS if all(k in m for m in first.values())]
    lines = [
        r"\begin{tabular}{ll" + "c" * len(columns) + "}",
        r"\toprule",
        "Env. & Method & " + " & ".join(h for _, h in columns) + r" \\",
        r"\midrule",
    ]
    for k, (env, comp) in enumerate(comparisons_by_env.items()):
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in comp]:
            cells = []
            for key, _ in columns:
                d, (_, scale, fmt) = comp[method][key], METRIC_MACRO[key]
                interval = _interval(scale * d["ci_lo"], scale * d["ci_hi"], fmt, signed=True)
                cells.append(
                    f"{_num(scale * d['diff'], fmt, signed=True)} \\scriptsize{{{interval}, $p_\\mathrm{{Holm}}$={_p(d['p_holm'])}}}"
                )
            lines.append(f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def long_table(estimates_by_env: dict, comparisons_by_env: dict) -> str:
    """Appendix: the training-budget check (twice the steps, new seeds)."""
    keys = ["sp", "xp", "gap", "perm_js"]
    header = (
        r"Env. & Method & Self-play $\uparrow$ & Cross-play $\uparrow$ & SP$-$XP gap $\downarrow$ & "
        r"Perm.\ JS ($\times 10^{-3}$) & $\Delta$ gap vs.\ IPPO \\"
    )
    lines = [r"\begin{tabular}{ll" + "c" * (len(keys) + 1) + "}", r"\toprule", header, r"\midrule"]
    for env, est in estimates_by_env.items():
        for method in [m for m in METHOD_ORDER if m in est]:
            cells = [_cell(est[method][key], key) for key in keys]
            d = comparisons_by_env.get(env, {}).get(method, {}).get("gap")
            cells.append("--" if d is None else f"{_num(d['diff'], signed=True)} \\scriptsize{{$p$={_p(d['p'])}}}")
            lines.append(f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def lambda_table(sweep) -> str:
    """Appendix: the lambda sweep on the selection seeds (validation episodes), with IPPO as lambda = 0."""
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Env. & $\lambda$ & Self-play $\uparrow$ & Cross-play $\uparrow$ & SP$-$XP gap & Perm.\ JS ($\times 10^{-3}$) \\",
        r"\midrule",
    ]
    for k, (env, rows) in enumerate(sweep.groupby("env", sort=False)):
        if k:
            lines.append(r"\midrule")
        for _, r in rows.sort_values("lam").iterrows():
            lam = "0 (IPPO)" if r["method"] == "ippo" else f"{r['lam']:g}"
            cells = [f"{_num(r[key])} \\scriptsize{{{_interval(r[f'{key}_lo'], r[f'{key}_hi'])}}}" for key in ("sp", "xp")]
            cells.append(_num(r["gap"]))
            cells.append(
                f"{_num(1e3 * r['perm_js'], '{:.1f}')} \\scriptsize{{{_interval(1e3 * r['perm_js_lo'], 1e3 * r['perm_js_hi'], '{:.1f}')}}}"
            )
            lines.append(f"{ENV_LABELS.get(env, env)} & {lam} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def compute_table(compute: dict) -> str:
    """Appendix: training wall-clock per run (median and range, minutes) by suite, environment and method."""
    lines = [
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Suite & Env. & Method & Runs & Minutes per run \\",
        r"\midrule",
    ]
    for suite in ("lambda", "main", "long"):
        if suite not in compute:
            continue
        for env, by_variant in sorted(compute[suite]["per_run_minutes"].items()):
            for variant, r in sorted(by_variant.items()):
                minutes = f"${r['median']:.1f}$ \\scriptsize{{$[{r['min']:.1f}, {r['max']:.1f}]$}}"
                method, _, lam = variant.partition("_lam")
                name = METHOD_LABELS.get(method, method) + (rf" $\lambda{{=}}{lam}$" if lam else "")
                lines.append(f"{suite} & {ENV_LABELS.get(env, env)} & {name} & ${int(r['count'])}$ & {minutes} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def markdown_table(estimates_by_env: dict) -> str:
    rows = [
        "| Env | Method | Self-play ↑ | Cross-play ↑ | SP−XP gap ↓ | Perm. JS (×10⁻³) | Order-shift drop |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(e, key):
        _, scale, f = METRIC_MACRO[key]
        return f"{f.format(scale * e.mean)} [{f.format(scale * e.lo)}, {f.format(scale * e.hi)}]".replace("-", "−")

    for env, est in estimates_by_env.items():
        for method in [m for m in METHOD_ORDER if m in est]:
            e = est[method]
            cells = [fmt(e[k], k) if k in e else "–" for k in ("sp", "xp", "gap", "perm_js", "shift_drop")]
            rows.append(f"| {ENV_LABELS.get(env, env)} | {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def overhead_table(overhead: dict) -> str:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Env. & Method & PPO update (s) & Inference forward (ms) & Actor parameters \\",
        r"\midrule",
    ]
    for k, (env, entry) in enumerate(overhead["envs"].items()):
        methods = entry["methods"]
        if k:
            lines.append(r"\midrule")
        for method in [m for m in METHOD_ORDER if m in methods]:
            r = methods[method]
            lines.append(
                f"{ENV_LABELS.get(env, env)} & {METHOD_LABELS[method]} & ${r['update_seconds']:.2f}$ (${r['update_vs_ippo']:.2f}\\times$) & "
                f"${r['forward_ms']:.3f}$ (${r['forward_vs_ippo']:.2f}\\times$) & ${r['actor_params']}$ \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def numbers_tex(estimates_by_env: dict, comparisons_by_env: dict, report: dict, long: tuple[dict, dict] | None = None) -> str:
    """LaTeX macros for every result quoted in the paper text (generated; do not edit).

    ``long`` holds the estimates and comparisons of the training-budget suite; their macros are prefixed ``Long``.
    """
    lines = ["% Generated by scripts/analyze.py from results/. Do not edit by hand."]

    def define(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    _estimate_macros(define, estimates_by_env, comparisons_by_env, report)
    if long is not None:
        _estimate_macros(define, *long, {}, prefix="Long")
    _report_macros(define, report)
    return "\n".join(lines) + "\n"


def _estimate_macros(define, estimates_by_env: dict, comparisons_by_env: dict, report: dict, prefix: str = "") -> None:
    for env, est in estimates_by_env.items():
        e_name = prefix + ENV_MACRO[env]
        for method, metrics in est.items():
            m_name = METHOD_MACRO[method]
            for key, (k_name, scale, fmt) in METRIC_MACRO.items():
                if key in metrics:
                    x = metrics[key]
                    define(f"{e_name}{m_name}{k_name}", _num(scale * x.mean, fmt))
                    define(f"{e_name}{m_name}{k_name}ci", _interval(scale * x.lo, scale * x.hi, fmt))
        for method, comp in comparisons_by_env.get(env, {}).items():
            m_name = METHOD_MACRO[method]
            for key, (k_name, scale, fmt) in METRIC_MACRO.items():
                if key not in comp:
                    continue
                d = comp[key]
                define(f"{e_name}{m_name}d{k_name}", _num(scale * d["diff"], fmt, signed=True))
                define(f"{e_name}{m_name}d{k_name}ci", _interval(scale * d["ci_lo"], scale * d["ci_hi"], fmt, signed=True))
                define(f"{e_name}{m_name}d{k_name}mde", _num(scale * d["mde"], fmt))
                define(f"{e_name}{m_name}d{k_name}p", _p(d["p"]))
                if "p_holm" in d:
                    define(f"{e_name}{m_name}d{k_name}ph", _p(d["p_holm"]))
        for method, rho in report.get("within_method_spearman_perm_js_vs_compatibility", {}).get(env, {}).items():
            define(f"{e_name}{METHOD_MACRO[method]}Rho", _num(rho))


def _report_macros(define, report: dict) -> None:
    for env, dec in report.get("additive_decomposition", {}).items():
        e_name = ENV_MACRO[env]
        define(f"{e_name}AddRsq", f"${100 * dec['r2_additive']:.1f}\\%$")
        define(f"{e_name}EgoPartnerCorr", _num(dec["corr_ego_partner_effect"]))
        define(f"{e_name}ResidSD", _num(dec["sd_residual"]))
        define(f"{e_name}TotalSD", _num(dec["sd_total"]))
        if "residual_split_half_corr" in dec:
            define(f"{e_name}ResidRel", _num(dec["residual_split_half_corr"]))
    for env, by_method in report.get("late_training_slope", {}).items():
        means = [s["mean"] for s in by_method.values()]
        define(f"{ENV_MACRO[env]}SlopeLo", _num(min(means)))
        define(f"{ENV_MACRO[env]}SlopeHi", _num(max(means)))
        define(f"{ENV_MACRO[env]}SlopePos", str(sum(s["n_positive"] for s in by_method.values())))
        define(f"{ENV_MACRO[env]}SlopeN", str(sum(s["n"] for s in by_method.values())))
    for env, by_method in report.get("seed_sd_self_play", {}).items():
        for method, sd in by_method.items():
            define(f"{ENV_MACRO[env]}{METHOD_MACRO[method]}SPsd", _num(sd))
    tuning = report.get("tuning", {})
    for key, name in [
        ("simple_spread_v3/3M/chosen", "TuneSpreadChosen"),
        ("simple_spread_v3/6M/chosen", "TuneSpreadLonger"),
        ("simple_adversary_v3/3M/chosen", "TuneAdversaryChosen"),
    ]:
        if key in tuning:
            define(name, _num(tuning[key]))
    others = [v for k, v in tuning.items() if k.startswith("simple_adversary_v3/3M/") and not k.endswith("/chosen")]
    if len(others) == 1:
        define("TuneAdversaryOther", _num(others[0]))
    for suite, record in report.get("compute", {}).items():
        define(f"Compute{suite.capitalize()}Hours", f"{record['total_hours']:.0f}")
    for env, entry in report.get("overhead", {}).get("envs", {}).items():
        for method, r in entry["methods"].items():
            define(f"{ENV_MACRO[env]}{METHOD_MACRO[method]}ActorParams", f"{r['actor_params']:,}".replace(",", "{,}"))
    for env, lam in report.get("selected_lambda", {}).items():
        define(f"{ENV_MACRO[env]}Lambda", f"{lam:g}")
    ref = report.get("reference_returns", {})
    for key, name in [("random", "RefRandom"), ("noop", "RefNoop"), ("assignment", "RefAssign")]:
        if key in ref:
            define(name, _num(ref[key]["mean"]))
    control = report.get("positive_control", {})
    for key, name in [("one_convention", "PCOne"), ("two_conventions", "PCTwo")]:
        if key in control:
            define(f"{name}Gap", _num(control[key]["gap"]))
            define(f"{name}Rsq", f"${100 * control[key]['r2_additive']:.1f}\\%$")
            define(f"{name}ResidRel", _num(control[key]["residual_split_half_corr"]))
    for key, name in [("first_teammate_landmark", "PCOrder"), ("greedy_west", "PCInvariant")]:
        if key in control:
            for kind in ("relabeling", "genuine", "resampled"):
                define(f"{name}{kind.capitalize()}", _num(control[key][f"change_{kind}"], signed=True))
    if "env_speed" in report:
        define("EnvSpeedup", f"{report['env_speed']['speedup']:.0f}")
