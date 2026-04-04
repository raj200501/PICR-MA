from __future__ import annotations


def to_pm(mean: float, se: float, prec: int = 3) -> str:
    return f"{mean:.{prec}f} $\\pm$ {se:.{prec}f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )
