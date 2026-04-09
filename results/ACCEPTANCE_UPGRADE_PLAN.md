# Acceptance Upgrade Plan

Plan date: 2026-04-08

## Chosen improvements

### 1. Matched IPPO-to-PICR mismatch analysis

Why chosen:

- This is the highest-leverage scientific upgrade available without new training.
- It answers the real reviewer question more directly than the older cross-method correlation framing.
- It tests what PICR-MA actually changes relative to its base learner under matched seeds and budgets.

Success looks like:

- A generated artifact with per-seed PICR-minus-IPPO deltas.
- A bootstrap summary showing whether the cross-play shift is stable or not.
- A main-paper figure and support table that make the mismatch claim more causal and less confounded.

Verification:

- `./.venv/bin/python scripts/make_tables.py`
- `./.venv/bin/python scripts/make_plots.py`
- `./.venv/bin/pytest -q`

### 2. Implementation trust and fairness upgrade

Why chosen:

- The NumPy learner is a real reviewer risk.
- A full rewrite is not worth the cost in this pass, but the repo can still document fairness and capacity much better.
- Clarifying that IPPO, PICR-MA, and Other-Play share the same MLP and budget reduces avoidable skepticism.

Success looks like:

- A generated support artifact with parameter counts, runtime ratios, and shared-backbone flags.
- Paper text that is explicit about same-family evaluation and matched-capacity comparisons.
- Analysis tests covering the new summary machinery.

Verification:

- `./.venv/bin/python scripts/make_tables.py`
- `./.venv/bin/python scripts/make_plots.py`
- `./.venv/bin/pytest -q`

### 3. Packaging and official-target recheck

Why chosen:

- The repo already has a real NeurIPS package, so the best move is to make its status correct and explicit.
- Official 2026 workshop availability can change, so this had to be rechecked.
- A clean rebuild and documented generic-workshop-ready status reduce avoidable submission risk.

Success looks like:

- An updated official-target memo.
- A `final.tex` that documents why workshop mode is not hard-coded yet.
- A successful rebuild of `final.pdf`.

Verification:

- Official NeurIPS 2026 source check on 2026-04-08
- `tectonic final.tex`

## What is intentionally left unchanged

- No new training campaign.
- No new major baseline.
- No PyTorch port.
- No named-workshop retargeting without an official public workshop page.
- No attempt to turn the paper back into a general algorithm-win story.
