# Workshop Review

## Summary

This paper studies PICR-MA, a policy-side permutation-consistency regularizer for decentralized MARL, and evaluates it on two MPE tasks with held-out partner cross-play, permutation sensitivity diagnostics, and lightweight ablations. The paper's strongest result is not that PICR-MA wins universally, but that it reliably lowers permutation sensitivity while yielding environment-dependent effects on zero-shot coordination. The adversary setting is a positive case; the spread setting is a counterexample; the new mismatch analysis is the main intellectual contribution.

## Strengths

- Honest and coherent reframing around the evidence actually available.
- Clean method with no inference-time architectural change.
- Real held-out partner evaluation instead of only in-team return.
- Stronger than a toy artifact package:
  - 5 seeds
  - 50k-step main cohort
  - pilot lambda sweep
  - dropout ablation
- Useful mechanistic diagnostic figure connecting invariance and ZSC.
- Official NeurIPS 2026 template adoption with a real compiled PDF.

## Weaknesses

- The learner is still a lightweight NumPy implementation, not a standard PyTorch PPO stack.
- Scale remains modest by conference standards.
- PICR-MA does not beat Other-Play or Deep Sets broadly enough to support a stronger method claim.
- The paper's value depends heavily on the reader appreciating the mixed-result lesson.

## Main reasons to accept

- The paper turns a mixed method result into a real empirical insight rather than hiding it.
- The method is simple and the diagnostic analysis is useful for future MARL symmetry work.
- The artifact package is reproducible, laptop-safe, and unusually transparent about failure cases.

## Main reasons to reject

- Some workshop reviewers may still see the empirical scale as too small.
- The novelty is incremental if judged only as a regularizer.
- Readers looking for a clear algorithmic win may find the result set underwhelming.

## Confidence

0.84
