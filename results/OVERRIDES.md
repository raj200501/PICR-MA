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
- `Deep Sets` is implemented as a compact mean-aggregated encoder for teammate blocks to stay CPU-friendly.

## Reduced scope forced by hardware / local setup

- Active profile: `PROFILE_LEAN_SAFE`
- Seeds: 2 per method/environment in Stage 1
- Stage 2 was not run.
  Reason: the lean-safe scope already produced a complete Stage 1 core matrix, while lambda sweeps and teammate-dropout ablations would expand runtime and artifact volume beyond the current target for this turn.

- LaTeX compilation was not completed in automation.
  Reason: `pdflatex` was not available in the local environment.
