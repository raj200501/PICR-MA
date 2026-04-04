# FINAL REPORT

## Active profile

- `PROFILE_LEAN_SAFE`

## What was implemented

- Explicit observation slicing with manual teammate-block definitions for MPE.
- PICR-MA symmetric KL regularizer with dual permuted views and optional teammate dropout hook.
- Shared-parameter NumPy on-policy actor--critic baselines for `ippo`, `picr_ma`, `deepsets`, and `otherplay`.
- Tiny 3-agent matrix coordination and stag-hunt environments.
- PettingZoo MPE training/evaluation wrappers for `simple_spread_v3` and `simple_adversary_v3`.
- Held-out partner cross-play evaluation from saved checkpoints.
- Permutation-sensitivity evaluation via Jensen--Shannon divergence.
- Aggregate CSV/JSON exports, publication-style PDF plots, LaTeX table fragments, and manuscript population.

## What actually ran

- Preflight and install check under the repo-local `.venv`
- Smoke runs
- Stage 0 runs
- Stage 1 core matrix:
  - environments: `matrix_coordination`, `matrix_staghunt`, `simple_spread_v3`, `simple_adversary_v3`
  - methods: `ippo`, `picr_ma`, `deepsets`, `otherplay`
  - seeds: 2
  - total Stage 1 runs: 32
- Cross-play evaluation for all Stage 1 runs
- Plot, table, artifact-inventory, and manuscript population passes

## What was skipped

- Stage 2 lambda sweep
- Stage 2 teammate-dropout ablation
- DQN baseline
- Automated LaTeX compilation

## Why anything was skipped

- Stage 2 was deferred to keep the run set within the lean-safe budget and because Stage 1 already completed end-to-end.
- DQN was unnecessary for completing the main PICR-MA package under current constraints.
- `pdflatex` was not installed locally.

## Runtime and disk

- Total recorded runtime across saved runs: about `54.896` seconds
- Stage 1 recorded runtime: about `51.461` seconds
- Generated `results/` footprint: about `4.0M`
- Generated `latex/` footprint: about `128K`
- Repo-local `.venv` footprint: about `241M`

## Exact reproduction commands

```bash
make install
make preflight
./.venv/bin/python scripts/install_check.py
./.venv/bin/python scripts/smoke_test.py
./.venv/bin/python scripts/run_stage0.py
./.venv/bin/python scripts/run_stage1.py
make plots
make tables
make paper
make clean_artifacts
make report
```

## What remains for human polishing

- Review the Stage 1 figures/tables and decide whether to keep the current lean-safe scope or run a Stage 2 ablation sweep.
- Install `pdflatex` and compile the paper locally.
- Tighten manuscript prose around the strongest and weakest environment-specific outcomes if preparing for submission.
- Optionally replace the NumPy learner with a PyTorch implementation once a stable local `torch` install is available.
