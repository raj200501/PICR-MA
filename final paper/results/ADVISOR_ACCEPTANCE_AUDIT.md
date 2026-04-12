# Advisor Acceptance Audit

Audit date: 2026-04-08

Scope note: this is the starting-point audit for the repo state before the upgrades in this pass were written into the paper.

## A. Current one-sentence paper thesis

Policy-level permutation consistency reliably lowers teammate-order sensitivity, but its effect on held-out coordination is environment-dependent, so invariance diagnostics are not reliable stand-ins for zero-shot coordination quality.

## B. Current strongest evidence

- A real 5-seed, 50k-step MPE cohort with IPPO, PICR-MA, Deep Sets, and Other-Play under matched budgets.
- Held-out partner evaluation from saved checkpoints instead of in-team return only.
- A mixed-result story that is actually informative:
  - PICR-MA lowers permutation sensitivity on both MPE tasks.
  - PICR-MA improves the cohort-average same-family cross-play number over IPPO on `simple_adversary_v3`.
  - PICR-MA loses clearly on `simple_spread_v3`.
- A 20k lambda sweep and teammate-dropout ablation that constrain the claim instead of flattering it.
- An existing compiled NeurIPS 2026 package with figures, tables, and transparent review artifacts.

## C. Current weakest evidence

- The learner is still a compact NumPy actor-critic rather than a standard PyTorch PPO stack.
- The main positive result is concentrated in one environment and does not establish broad superiority over Other-Play or Deep Sets.
- The evaluation emphasis is same-family held-out cross-play; broader cross-family gains are weaker.
- The overall benchmark scale is modest: two MPE tasks, 50k steps, laptop compute.
- The older mismatch framing leaned on a weak correlation story rather than the strongest matched comparison.

## D. Top 5 likely reasons a NeurIPS workshop reviewer would reject this paper

- The implementation stack looks non-standard enough that some reviewers will worry the result is entangled with learner quirks.
- The paper does not offer a broad algorithmic win; on `simple_spread_v3` PICR-MA is simply worse.
- The empirical scope is small for a method paper: two main environments and one partially positive setting.
- The positive adversary result is not obviously robust across partner families or matched seeds.
- The package was still generic-theme targeted rather than tied to a named official 2026 workshop.

## E. Top 5 reasons a well-matched workshop might accept it

- The paper is honest about mixed results and turns them into a useful empirical lesson instead of overselling.
- The core question is workshop-appropriate: when does a symmetry bias help zero-shot coordination, and when does it mislead?
- The evaluation stack is better than a toy result:
  - 5 seeds
  - held-out partner cross-play
  - permutation diagnostics
  - ablations
  - runtime accounting
- The method is simple enough to be reusable by other MARL researchers who already have MLP policies.
- The repo already contains unusually strong transparency artifacts for a small-compute study.

## F. Strongest paper identity

Primary framing: `invariance-versus-coordination mismatch paper`

Justification:

- `simple regularizer paper` is too weak because the method does not dominate.
- `zero-shot coordination evaluation paper` is close, but it understates the paper's mechanistic point.
- `MARL symmetry diagnostic paper` is plausible, but still broader than the actual strongest contribution.
- The most defensible identity is the mismatch claim itself:
  - PICR-MA moves the permutation-sensitivity diagnostic in the intended direction.
  - The coordination outcome does not move with the same reliability.
  - That mismatch is what makes the mixed evidence scientifically useful instead of merely disappointing.

## G. Realistic workshop recommendation score

`borderline`

Reason:

- The repo already clears the bar for a serious workshop submission.
- It does not yet clear the bar for an easy accept because the implementation and empirical scope still create obvious reviewer attack points.

## Ranked improvement shortlist

### 1. Replace the weak correlation-style mismatch framing with a matched IPPO-to-PICR delta analysis

- Expected acceptance impact: high
- Expected scientific value: high
- Expected compute cost: minimal
- Expected implementation risk: low
- Story effect: strengthens the core story

### 2. Add a direct implementation-trust and fairness upgrade

- Expected acceptance impact: medium-high
- Expected scientific value: medium
- Expected compute cost: negligible
- Expected implementation risk: low
- Story effect: strengthens the core story by reducing avoidable skepticism

### 3. Recheck official NeurIPS 2026 workshop status and clean up the workshop-ready package explicitly

- Expected acceptance impact: medium
- Expected scientific value: low
- Expected compute cost: negligible
- Expected implementation risk: low
- Story effect: mostly packaging, but important packaging

## Improvements intentionally not selected

- No broad new sweep: too expensive relative to likely acceptance gain.
- No major new baseline: high scope, weak probability of changing the conclusion.
- No PyTorch port in this pass: high-risk rewrite that would dominate the schedule.
- No named workshop retargeting without an official public target: would require invention.
