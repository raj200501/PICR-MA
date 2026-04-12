# Improvement Trial: paired_mismatch_analysis

## What changed

- Added paired PICR-MA versus IPPO delta exports:
  - `results/aggregate/paired_picr_vs_ippo.csv`
  - `results/aggregate/paired_picr_vs_ippo_stats.json`
- Replaced the old mismatch figure content with a matched-seed delta figure:
  - `latex/figures/invariance_coordination_mismatch.pdf`
- Added an appendix support table:
  - `latex/tables/paired_delta_results.tex`
- Rewrote the corresponding paper text to use the paired evidence.

## What was run

- `./.venv/bin/python scripts/make_tables.py`
- `./.venv/bin/python scripts/make_plots.py`
- `./.venv/bin/pytest -q`

## What evidence was produced

- On `simple_spread_v3`, PICR-MA lowers JS in `5/5` matched pairs and loses same-family cross-play in `5/5`.
- On `simple_adversary_v3`, PICR-MA lowers JS in `4/5` matched pairs and improves same-family cross-play in `3/5`.
- The spread cross-play delta has a fully negative bootstrap interval.
- The adversary cross-play delta is positive on average but still wide.

## Did it help?

Yes.

This is a better workshop argument than the older cross-method correlation framing because it isolates what the regularizer changes relative to its own base learner.

## Should it be included in the final paper?

Yes.

## Did it change the core story?

Yes, but by sharpening it rather than replacing it.

The story is now:

- the diagnostic shift is real,
- the coordination shift is less stable,
- the mismatch is visible in matched comparisons, not only in aggregate scatter.
