# FINAL REPORT

## Active profile

- `PROFILE_LEAN_SAFE`
- branch: `icml_hardening_pass`
- main approval: blocked

## What was implemented in the hardening pass

- Cohort-scoped partner-pool evaluation to remove cross-play contamination.
- Episode-consistent Other-Play teammate-order randomization.
- 20k pilot cohort across all main baselines on both MPE tasks.
- 20k lambda sweep for PICR-MA over `0.02`, `0.05`, `0.10`.
- 20k teammate-dropout ablation at dropout probability `0.10`.
- 50k main MPE cohort with 5 seeds per method and tuned PICR-MA (`lambda=0.02`).
- Updated manuscript text and regenerated figures/tables from the hardened cohort.
- Reviewer-style audit, self-review, and go/no-go documents.

## What actually ran

- Re-evaluation of all tracked `stage1` runs with the fixed partner-pool logic.
- `hardpilot20k`: 24 runs
- `lambda20k`: 18 runs
- `drop20k_p010`: 6 runs
- `hardmain50k`: 40 runs
- Full held-out cross-play evaluation for all of the above cohorts.
- Regenerated LaTeX tables and figures from the 50k main cohort.
- Full test suite: `14 passed`

## What was skipped

- Any merge or fast-forward into `main`
- Any push to a remote
- A PyTorch port of the learner
- Larger-than-50k full-baseline MPE sweeps
- Full appendix build-out and LaTeX compilation

## Why anything was skipped

- `main` is blocked because the project does not pass the internal ICML bar; see `results/ICML_GO_NO_GO.md`.
- A PyTorch port would have been a larger rewrite than was justified before resolving the empirical question.
- The 50k cohort already showed a mixed claim; larger sweeps were unlikely to rescue the project enough for a PASS under the remaining time and hardware budget.
- `pdflatex` was not available locally.

## Runtime and disk

- Recorded training runtime from the hardening cohorts:
  - `hardpilot20k`: `777.489` s
  - `lambda20k`: `1017.618` s
  - `drop20k_p010`: `100.225` s
  - `hardmain50k`: `1655.531` s
- Total recorded training runtime for the hardening cohorts: `3550.863` s
- Current `results/` footprint: about `31M`
- Current `latex/` footprint: about `128K`
- Free disk at end of pass: about `4.8 GiB`

## Exact reproduction commands

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/pytest -q

for env in simple_spread_v3 simple_adversary_v3; do
  for method in ippo picr_ma deepsets otherplay; do
    for seed in 0 1 2 3 4; do
      if [ "$method" = "picr_ma" ]; then
        PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/python scripts/train.py \
          --env "$env" --method "$method" --seed "$seed" --steps 50000 --tag hardmain50k \
          --max-cycles 25 --picr-lambda 0.02 --eval-episodes 16
      else
        PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/python scripts/train.py \
          --env "$env" --method "$method" --seed "$seed" --steps 50000 --tag hardmain50k \
          --max-cycles 25 --eval-episodes 16
      fi
    done
  done
done

for run in $(find results -maxdepth 1 -type d -name 'run_hardmain50k_*' | sort); do
  PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/python scripts/evaluate.py \
    --run-dir "$run" --results-dir results --partner-run-tag hardmain50k --episodes 16
done

PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/python scripts/make_tables.py
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ./.venv/bin/python scripts/make_plots.py
```

## What remains for human polishing

- Decide whether to invest in a PyTorch PPO port and substantially larger full-baseline runs.
- Build a real appendix package if the project is revived.
- Install `pdflatex` and compile the paper locally.
- Revisit the method claim entirely if future work still shows mixed environment dependence.
