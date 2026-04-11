# NeurIPS Workshop Positioning

## Final framing

Chosen framing:

`When Does Policy-Level Permutation Consistency Help Zero-Shot Coordination in Small MARL? A Controlled Study with PICR-MA`

This is no longer framed as a main-track claim that PICR-MA is a uniformly better default method.

## Why this framing is workshop-appropriate

- The method is technically clean and easy to understand.
- The experiments are careful and honest, but still laptop-scale.
- The headline result is mixed in a scientifically useful way.
- The paper contains a concrete empirical lesson:
  lowering permutation sensitivity is not by itself a sufficient proxy for better held-out coordination.

That is a strong workshop contribution even though it is not a strong main-track ``new default method'' contribution.

## What the paper now contributes

1. A simple policy-side permutation-consistency regularizer that leaves inference unchanged.
2. A controlled evaluation stack for held-out partner cross-play and teammate-order sensitivity.
3. A concrete positive case:
   PICR-MA improves IPPO on `simple_adversary_v3`.
4. A concrete counterexample:
   PICR-MA lowers JS but still loses on `simple_spread_v3`.
5. A sharper empirical conclusion:
   invariance diagnostics and coordination outcomes can diverge in small MARL.
