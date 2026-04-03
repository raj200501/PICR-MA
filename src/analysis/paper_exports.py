from pathlib import Path


def populate_main_tex(main_tex='main.tex'):
    p = Path(main_tex)
    t = p.read_text()
    old = "\\section{Results}\n% Intentionally left blank per request. Populate with curves/tables above."
    new = (
        "\\section{Results}\n"
        "\\input{latex/tables/main_results.tex}\n"
        "\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{latex/figures/learning_curves_mpe.pdf}"
        "\\caption{Training summary from available runs.}\\end{figure}"
    )
    t = t.replace(old, new)
    p.write_text(t)
