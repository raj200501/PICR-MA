# Workshop Target Selection

Primary target: theme-family target, not a named 2026 workshop

## Selected target family

Multi-agent RL / cooperative AI / zero-shot coordination

## Why this is the right fit

- The paper's core task is held-out partner zero-shot coordination in MARL.
- The method is about teammate-order symmetry and coordination robustness, not about generic single-agent RL.
- The strongest evidence is environment-dependent coordination behavior under different symmetry biases.

## Why this is not a named workshop

During the official-source audit, no public 2026 workshop list or workshop CFP pages were located on neurips.cc. Under that constraint, selecting a theme family is more honest than inventing a specific workshop title.

## How the paper was adapted to this target

- The title is now question-driven rather than main-track-claim-driven.
- The abstract foregrounds the controlled-study result:
  - PICR-MA lowers permutation sensitivity
  - helps on `simple_adversary_v3`
  - hurts on `simple_spread_v3`
  - lower permutation sensitivity does not reliably imply better ZSC
- The main-paper figures emphasize:
  - held-out partner ZSC
  - permutation sensitivity
  - the invariance-vs-coordination mismatch
- The paper no longer claims that PICR-MA is a new default method.

## What would change once a public workshop CFP exists

- If the target workshop is specifically MARL/cooperative-AI focused:
  - keep the current framing almost unchanged.
- If the target workshop is RL-foundations/generalization focused:
  - increase emphasis on diagnostic mismatch and environment dependence.
- If the target workshop is broader empirical-ML diagnostics:
  - foreground the ``metric mismatch'' lesson even more strongly.
