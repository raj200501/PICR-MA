# PICR Safe Advisor Audit

Timestamp: 2026-04-10 17:45:57 EDT

Scope: conservative do-no-harm audit of the current protected baseline package before any candidate-only edits.

## A. Current one-sentence paper thesis

PICR-MA reliably reduces teammate-order sensitivity, but the resulting zero-shot coordination effect is environment-dependent, so invariance diagnostics are not reliable stand-ins for held-out coordination quality.

## B. Current strongest evidence

- A real 5-seed, 50k-step MPE cohort for `simple_spread_v3` and `simple_adversary_v3`.
- Held-out partner cross-play from saved checkpoints instead of self-play return alone.
- A matched IPPO versus PICR-MA paired analysis that isolates what the regularizer changes relative to its own base learner.
- A transparent mixed-result story:
  - PICR-MA lowers permutation sensitivity on both MPE tasks.
  - PICR-MA improves the same-family cohort mean over IPPO on `simple_adversary_v3`.
  - PICR-MA loses clearly on `simple_spread_v3`.
- Supporting artifact discipline for runtime, parameter-count fairness, ablations, and workshop packaging.

## C. Current weakest evidence

- The learner is still a compact NumPy actor-critic rather than a standard PyTorch PPO stack.
- The main positive result is narrow:
  - one of two main MPE environments
  - concentrated in same-family partners
  - unstable in paired-seed comparisons
- The benchmark scope remains small for a method paper.
- Other-Play remains at least competitive on the stronger environment and better on `simple_spread_v3`.
- The package is generic-workshop-ready rather than tied to a specific official workshop target.

## D. Strongest current honest paper identity

Primary identity: `invariance-versus-coordination mismatch paper`

Justification:

- `simple regularizer paper` is too ambitious because PICR-MA does not dominate the baselines.
- `zero-shot coordination evaluation paper` is close, but it undersells the mechanistic point.
- `invariance diagnostic paper` captures part of the story, but the paper is strongest precisely because the diagnostic improvement and the coordination outcome come apart.
- The most defensible identity is the mismatch itself:
  - the regularizer reliably shifts the diagnostic,
  - the coordination effect is materially less stable,
  - the negative and mixed results are informative rather than accidental.

## E. Top 5 likely workshop rejection reasons

1. The implementation stack is non-standard enough that some reviewers will distrust external validity.
2. The paper is not a broad algorithmic win and loses clearly on `simple_spread_v3`.
3. The empirical scope is modest for a method submission.
4. The positive `simple_adversary_v3` signal is narrow and not cleanly robust across partner families or paired seeds.
5. The front matter still spends some attention on matrix-game sanity checks, which can make the paper look smaller than its strongest MPE evidence.

## F. Top 5 likely workshop acceptance reasons

1. The paper is honest about mixed results and turns them into a useful empirical lesson.
2. The core question is workshop-appropriate: when does a symmetry bias improve coordination, and when does it mislead?
3. The evaluation stack is substantially stronger than a toy study:
   same-family held-out cross-play, permutation diagnostics, paired deltas, ablations, and runtime accounting.
4. The method is simple enough to reuse in existing MLP-based MARL codebases.
5. The repo contains unusually strong transparency artifacts for a laptop-scale workshop submission.

## G. Realistic current recommendation

Recommendation for a well-matched NeurIPS workshop: `weak accept`

Reason:

- The current paper already clears the bar for a serious workshop submission.
- It still has obvious reviewer attack points, so this is not honestly a `strong accept`.
- The strongest case is for a workshop that values MARL generalization, robustness, or coordination diagnostics rather than universal method wins.

## H. Ranked list of the top 1-3 improvements that might genuinely help without risking the paper

### 1. Front-matter focus correction

Change:

- Keep the science unchanged, but revise the abstract and contribution framing so the 5-seed, 50k-step MPE cohort is unambiguously the main evidence and the matrix games are clearly appendix-only sanity checks.

Assessment:

- Expected scientific value: medium
- Expected acceptance impact: medium
- Expected compute cost: none
- Expected implementation risk: low
- Risk of making the paper worse: low
- Reversible: yes
- Would the paper still be stronger if the improvement fails: yes, because it can be reverted with no scientific cost

### 2. Title-level mismatch sharpening

Change:

- Consider a slightly sharper title that signals the paper is a controlled mismatch study rather than a broad new default method.

Assessment:

- Expected scientific value: low
- Expected acceptance impact: low-to-medium
- Expected compute cost: none
- Expected implementation risk: low-to-medium
- Risk of making the paper worse: medium, because over-correcting the title can make the paper sound smaller or more negative than necessary
- Reversible: yes
- Would the paper still be stronger if the improvement fails: yes, if rejected and reverted

No other candidate improvement currently clears the do-no-harm bar.

- No new experiments: too much compute, too much variance, too much risk.
- No baseline expansion: likely scope creep with uncertain payoff.
- No implementation rewrite: the PyTorch objection is real, but a rewrite is far too risky for this pass.
- No workshop-mode switch without an official target: unsupported packaging is worse than generic packaging.
