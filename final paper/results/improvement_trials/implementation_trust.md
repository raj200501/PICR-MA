# Improvement Trial: implementation_trust

## What changed

- Added a generated implementation/fairness artifact:
  - `results/aggregate/implementation_summary.csv`
- Added tests for the new statistical summary path:
  - `tests/test_statistics.py`
- Expanded the paper text to clarify:
  - same-family evaluation in the main table
  - shared MLP capacity for IPPO/PICR-MA/Other-Play
  - Deep Sets as the only capacity-changing baseline

## What was run

- `./.venv/bin/python scripts/make_tables.py`
- `./.venv/bin/python scripts/make_plots.py`
- `./.venv/bin/pytest -q`

## What evidence was produced

- IPPO, PICR-MA, and Other-Play all use the same cooperative parameter count in each MPE environment.
- Deep Sets alone changes inference-time capacity.
- PICR-MA runtime overhead is explainable as extra training-time forward passes, not a larger deployed model.
- The expanded analysis path is covered by a passing test suite.

## Did it help?

Yes.

It does not eliminate the NumPy-learner risk, but it reduces reviewer suspicion that the comparison is unfair or capacity-confounded.

## Should it be included in the final paper?

Yes, in concise form.

## Did it change the core story?

No.

It strengthens trust in the existing story.
