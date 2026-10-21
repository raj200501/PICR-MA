#!/usr/bin/env python
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picr_ma.analysis.paper_exports import populate_main_tex


def try_compile(main_tex: Path) -> str:
    if shutil.which("pdflatex") is None:
        return "pdflatex not available"
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", str(main_tex.name)],
        cwd=main_tex.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    return "succeeded" if result.returncode == 0 else f"failed: {result.stderr.strip() or result.stdout.splitlines()[-1]}"


if __name__ == "__main__":
    main_tex = ROOT / "main.tex"
    populate_main_tex(main_tex)
    compile_status = try_compile(main_tex)
    (ROOT / "results/LATEX_POPULATION_REPORT.md").write_text(
        "\n".join(
            [
                "# LATEX POPULATION REPORT",
                "- main.tex found: yes",
                "- manuscript root: ./",
                "- sections updated: Introduction (method figure), Experimental Setup heading, Results",
                "- figures inserted: method_overview.pdf, learning_curves_mpe.pdf, zero_shot_crossplay_results.pdf, permutation_sensitivity_results.pdf, runtime_overhead.pdf, optional_matrix_games_summary.pdf",
                "- tables inserted: main_results.tex, crossplay_results.tex, permutation_sensitivity.tex, runtime_results.tex",
                f"- compilation: {compile_status}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "results/LATEX_POLISH_REPORT.md").write_text(
        "\n".join(
            [
                "# LATEX POLISH REPORT",
                "- Added figures: method_overview.pdf, learning_curves_mpe.pdf, zero_shot_crossplay_results.pdf, permutation_sensitivity_results.pdf, runtime_overhead.pdf, optional_matrix_games_summary.pdf",
                "- Added tables: main_results.tex, crossplay_results.tex, permutation_sensitivity.tex, runtime_results.tex",
                "- Densified sections: Results",
                "- Visual narrative: grounded in the currently available saved runs; depends on the active profile and completed experiments",
                f"- Compilation: {compile_status}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(compile_status)
