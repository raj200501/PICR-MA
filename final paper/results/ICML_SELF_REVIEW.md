# ICML Self-Review

## Summary

This submission studies PICR-MA, a permutation-consistency regularizer for shared-parameter decentralized MARL. The paper's main evidence is a 5-seed, 50k-step MPE cohort on `simple_spread_v3` and `simple_adversary_v3`, plus 20k pilot ablations over the regularization weight and teammate dropout. The method is lightweight relative to architectural invariance baselines, and the repo is reasonably reproducible. Empirically, however, the story is mixed: PICR-MA improves held-out partner cross-play and lowers JS sensitivity on `simple_adversary_v3`, but underperforms IPPO, Deep Sets, and Other-Play on `simple_spread_v3`.

## Strengths

- The core method is simple, easy to implement, and architecture-agnostic.
- The hardening pass fixed a real evaluation bug: cross-play is now cohort-scoped rather than contaminated by all saved runs.
- The benchmark suite includes held-out partner cross-play, permutation sensitivity, lambda sweeps, and a teammate-dropout ablation.
- The main MPE evidence is materially stronger than the initial toy scaffold: 5 seeds and 50k steps per seed.
- Reporting is now much more honest than before; the manuscript no longer claims broad gains that are unsupported by the runs.

## Weaknesses

- The headline empirical claim is not robust across the main MPE tasks.
- PICR-MA is the weakest method on `simple_spread_v3` held-out same-family cross-play in the 50k cohort.
- The mechanistic story is weaker than the method pitch suggests: lower permutation sensitivity does not monotonically correspond to better cross-play.
- The best-performing PICR-MA setting in the 20k lambda sweep (`lambda=0.02`) does not improve JS over IPPO on the pilot environments.
- Other-Play remains a very strong baseline and is at least as strong as PICR-MA on `simple_adversary_v3` while clearly stronger on `simple_spread_v3`.
- The implementation still uses a lightweight NumPy learner rather than a more standard PyTorch PPO stack.
- The experimental scale, while much improved, is still modest for a serious conference submission.

## Main reasons to accept

- The method is simple and genuinely practical to bolt onto existing decentralized learners.
- The adversary-side result is interesting: PICR-MA beats IPPO and Deep Sets on held-out partner cross-play while reducing permutation sensitivity.
- The paper would be valuable as a careful negative-or-mixed result if positioned as a tradeoff study rather than a broad win.
- The artifact is reproducible, laptop-safe, and transparent about its limitations.

## Main reasons to reject

- There is no strong, clean headline claim that survives both MPE environments.
- The strongest environment-level result is not against the strongest baseline: Other-Play slightly exceeds PICR-MA on adversary cross-play.
- The method loses clearly on `simple_spread_v3`, which directly undermines any claim that PICR-MA is a strong new default.
- The invariance diagnostic does not provide a convincing mechanistic bridge to ZSC gains.
- The remaining scale and learner-credibility limitations make the evidence easier to dismiss than it should be.

## Questions for authors

1. Would the same mixed pattern persist under a standard PyTorch PPO implementation and longer training horizons?
2. Why does the best cross-play lambda in the 20k sweep not improve JS over IPPO, while higher lambdas improve JS but hurt coordination?
3. Is `simple_spread_v3` exposing a fundamental failure mode for the consistency regularizer, or only an optimization issue?
4. Can the method be reformulated to regularize only the subsets of teammate features that actually induce harmful asymmetries?
5. Why does Other-Play retain stronger spread performance despite higher permutation sensitivity?

## Reproducibility assessment

- Code availability: good
- Environment setup: good
- Determinism and seeds: good
- Evaluation protocol clarity: improved and now acceptable
- Artifact honesty: good
- Remaining reproducibility concern: the custom NumPy learner is nonstandard, which limits external confidence even though the code is easy to run

## Confidence level

High. The hardening pass materially strengthened the evidence and still failed to produce a robust win story.
