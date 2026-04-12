# OVERRIDES

## Scientific and implementation deviations

- Backend deviation: the current implementation is NumPy-based rather than PyTorch-based.
  Reason: `torch` was not available in the local environment, while PettingZoo/MPE and the scientific core were still feasible under a lighter dependency footprint.

- Base learner deviation: the repo currently uses a lightweight shared-parameter clipped on-policy actor--critic rather than a PyTorch PPO stack.
  Reason: resource safety and local dependency constraints on the M1 Mac.

- DQN was not implemented.
  Reason: the main method and all completed experiments were already covered by the on-policy path, and adding a second learner would have increased scope without being necessary for Stage 0/Stage 1 honesty.

## Environment-slicing assumptions

- `simple_spread_v3` uses manual teammate blocks:
  - teammate block 1: indices `[10, 11, 14, 15]`
  - teammate block 2: indices `[12, 13, 16, 17]`
  Reason: PettingZoo observation order is `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`, so teammate-specific position and communication features are grouped manually for permutation.

- `simple_adversary_v3` is instantiated with `N=3` good agents rather than the default `N=2`.
  Reason: with only two good agents, each agent has only one teammate, making teammate-order permutation trivial. Using `N=3` yields two cooperative teammate blocks and makes PICR-MA meaningful.

- `simple_adversary_v3` good-agent slicing treats the adversary-relative position as part of the non-teammate feature set and permutes only the good-teammate blocks at the tail of the observation.

- Matrix games are implemented as 3-agent one-step games built from pairwise 2x2 coordination/stag-hunt payoffs.
  Reason: 2-agent games have only one teammate and cannot express teammate-order permutations.

## Baseline simplifications

- `Other-Play` is implemented as lightweight teammate-order randomization during training rather than a larger environment-symmetry framework.
- During the hardening pass, `Other-Play` was updated to use episode-consistent teammate-order randomization rather than resampling every step.
- `Deep Sets` is implemented as a compact mean-aggregated encoder for teammate blocks to stay CPU-friendly.

## Reduced scope forced by hardware / local setup

- Active profile: `PROFILE_LEAN_SAFE`
- Main hardening cohort: 5 seeds per method/environment at 50k MPE steps.
- The full hardening pass did run a lambda sweep and teammate-dropout ablation, but only at 20k pilot scale.
  Reason: this was sufficient to establish method sensitivity without committing to much larger ablation budgets on the constrained machine.
- The project was not scaled to 250k+ full-baseline MPE runs.
  Reason: after the 50k, 5-seed main cohort, the evidence was already mixed enough that further scale was unlikely to change the internal PASS/FAIL decision proportionally to the extra runtime.
- PICR-MA main-cohort lambda was changed from the original `0.2` scaffold default to `0.02`.
  Reason: the 20k lambda sweep showed that the default setting was too destructive; `0.02` was the least bad setting for held-out partner cross-play.
- Teammate dropout is not part of the main method story.
  Reason: the `drop20k_p010` ablation hurt held-out partner cross-play on both MPE tasks.
- Main approval remains blocked.
  Reason: the 50k, 5-seed evidence supports only a mixed tradeoff claim and does not clear the internal ICML bar.

- LaTeX compilation was not completed in automation.
  Reason: `pdflatex` was not available in the local environment.
