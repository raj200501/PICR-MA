# Paper

LaTeX source for *Does Teammate-Order Invariance Help Zero-Shot Coordination?
A Controlled Study in Cooperative MARL*.

**Status:** being rewritten around the corrected experiments (see
[../docs/corrections.md](../docs/corrections.md)). The related-work, method,
protocol and appendix sections are in `sections/`; the abstract, introduction,
results and discussion are added once the experiments finish, so the paper
does not build yet.

Every table and figure, and every estimate, interval and p-value quoted in the
text, is generated from `results/` by `scripts/analyze.py` (into `figures/`
and `tables/`, including the macros in `tables/numbers.tex`); the test suite
checks that every macro the text uses is defined. Protocol constants (seeds,
step budgets, hyperparameters) are typed in the text and match
`configs/experiments.yaml`. The method figure is drawn by
`scripts/make_method_figure.py`.

Build with `make paper` from the repository root. It needs TeX Live
(`latexmk`, `pdflatex`, `natbib`) and works from the committed `results/`, with
no training or evaluation.
