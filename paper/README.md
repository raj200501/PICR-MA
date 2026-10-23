# Paper

LaTeX source for *Does Teammate-Order Invariance Help Zero-Shot Coordination?
A Controlled Study in Cooperative MARL*.

**Status:** being rewritten around the corrected experiments (see
[../docs/corrections.md](../docs/corrections.md)). The related-work, method and
protocol sections are in `sections/`; the abstract, introduction, results and
discussion are added once the main suite finishes, so the paper does not build
yet.

Every table, figure and number in the text is generated from `results/` by
`scripts/analyze.py` (into `figures/` and `tables/`, including
`tables/numbers.tex`). Build with `make paper` from the repository root.
