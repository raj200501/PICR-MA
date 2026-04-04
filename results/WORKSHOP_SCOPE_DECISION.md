# Workshop Scope Decision

No new training runs were launched for the workshop pivot.

## Why no new experiments were necessary

- The current repo already contains the evidence needed for the workshop claim:
  - a 5-seed, 50k-step main MPE cohort (`hardmain50k`)
  - a 20k lambda sweep (`lambda20k_*`)
  - a 20k teammate-dropout ablation (`drop20k_p010`)
  - repaired held-out partner evaluation and cross-play matrices
- The workshop framing is about the relationship between permutation sensitivity and zero-shot coordination, not about squeezing out one more positive benchmark win.
- Larger new sweeps would have increased compute cost without addressing the main scientific question as directly as better analysis and manuscript framing.

## What was added instead of more training

- A new main-paper figure:
  - `latex/figures/invariance_coordination_mismatch.pdf`
- A compact main-paper zero-shot summary figure:
  - `latex/figures/zero_shot_crossplay_summary.pdf`
- A workshop-oriented ablation table:
  - `latex/tables/ablation_results.tex`
- A full NeurIPS 2026 workshop-ready manuscript build:
  - `final.tex`
  - `final.pdf`

## Why larger sweeps were not needed

- The central workshop insight was already visible in the existing runs:
  - PICR-MA lowers permutation sensitivity.
  - PICR-MA helps on `simple_adversary_v3`.
  - PICR-MA loses on `simple_spread_v3`.
  - Lower permutation sensitivity does not reliably imply better held-out coordination.
- The workshop bar is insight, honesty, and empirical care, not main-track scale.
- The repo is still laptop-constrained and deliberately lean-safe; preserving reproducibility and machine safety was more important than chasing another partial scale increase.
