# PICR-MA

PICR-MA is a small-compute research repo for permutation-invariant multi-agent reinforcement learning and zero-shot coordination. The core method adds a symmetric KL consistency penalty between two independently permuted teammate-order views of the same observation, so a policy is discouraged from using arbitrary teammate ordering as a shortcut.

## What PICR-MA is

- Base idea: keep the base learner unchanged at inference time and regularize only during training.
- Training penalty: for teammate feature blocks `U = [u_1, ..., u_{n-1}]`, sample two permutations, run the policy twice, and add `0.5 * KL(p1 || p2) + 0.5 * KL(p2 || p1)`.
- Goal: improve order invariance and zero-shot coordination without requiring a custom invariant architecture.

## Why order invariance matters in MARL

In many cooperative tasks the order of teammate features in the observation vector is arbitrary. A vanilla MLP still treats the first teammate block differently from the second. That spurious asymmetry can hurt held-out partner coordination and inflate permutation sensitivity even in fully symmetric tasks.

## Environments

- PettingZoo MPE `simple_spread_v3`
- PettingZoo MPE `simple_adversary_v3` with `N=3` good agents so teammate permutations are non-trivial
- Tiny 3-agent one-step matrix coordination and stag-hunt-style games built to expose irrelevant teammate-order features cheaply

## Methods and baselines

- `ippo`: shared-parameter decentralized on-policy actor-critic
- `picr_ma`: `ippo` plus the PICR-MA symmetric KL regularizer
- `deepsets`: permutation-invariant Deep Sets encoder baseline
- `otherplay`: lightweight teammate-order randomization baseline

## Zero-shot coordination protocol

- Each run saves a checkpointed policy under `results/run_*/best.ckpt`.
- `scripts/evaluate.py` loads held-out partner checkpoints from the local partner pool and writes `eval_crossplay.csv.gz` for each run.
- Cross-play is checkpoint-based and excludes the exact same run from the held-out pool.

## Permutation sensitivity metric

- `src/eval/permutation_sensitivity.py` measures Jensen-Shannon divergence between action distributions under teammate-order permutations.
- Per-run samples are saved to `eval_permutation_sensitivity.csv.gz`.
- Aggregate summaries are exported to `results/aggregate/` and LaTeX-ready tables under `latex/tables/`.

## Setup and commands

Create the local environment and install dependencies:

```bash
make install
```

Run preflight and choose the active safety profile:

```bash
make preflight
```

Run the smoke suite:

```bash
make smoke
```

Run the Stage 1 core matrix:

```bash
make stage1
```

Regenerate figures and tables:

```bash
make plots
make tables
```

Populate the manuscript from real artifacts:

```bash
make paper
```

Refresh artifact accounting and print repo status:

```bash
make clean_artifacts
make report
```

## Manuscript population

- `main.tex` is treated as the manuscript root.
- Plot exports go to `latex/figures/`.
- LaTeX table fragments go to `latex/tables/`.
- `scripts/make_paper.py` rewrites the results section to reference generated artifacts and attempts a local `pdflatex` compile if available.

## Disk and runtime safety

- Active profile is written to `results/ACTIVE_PROFILE.txt`.
- Artifacts are tracked in `results/artifact_inventory.csv`.
- Runs are sequential and small by default.
- The repo-local `.venv` currently uses about a few hundred MB; generated result artifacts are much smaller than the configured budgets under typical smoke and lean-safe runs.

Expected generated-artifact ceilings by profile:

- `PROFILE_FULL_SAFE`: up to 3.0 GB
- `PROFILE_LEAN_SAFE`: up to 1.5 GB
- `PROFILE_ULTRA_LEAN`: up to 700 MB
- `PROFILE_SCAFFOLD_ONLY`: up to 250 MB

## Known limitations

- The current implementation is NumPy-based rather than PyTorch-based. This keeps the repo lighter on an M1 Mac but is a scientific/runtime deviation that is recorded in `results/OVERRIDES.md`.
- Cross-play summaries only become meaningful once multiple compatible checkpoints exist for the same environment.
- Manuscript quality depends on the runs actually present in `results/`; empty partner pools are reported as unavailable rather than guessed.
