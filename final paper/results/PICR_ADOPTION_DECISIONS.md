# PICR Adoption Decisions

Timestamp: 2026-04-10 17:50 EDT

## Attempted changes

### 1. Front-matter focus correction

Decision: `adopted`

Reason:

- The change improves narrative sharpness and workshop fit by making the evidence hierarchy explicit:
  the MPE cohort is the main evidence and the matrix games are appendix-only sanity checks.
- It does not weaken the paper's honest identity.
- It introduces no unsupported claim.
- It required no new computation.
- It compiled cleanly inside the candidate package.
- The side-by-side diff is narrow and reversible.

## Changes not attempted

- Title rewrite:
  rejected at prefilter stage because it was not clearly safer or higher-value than the front-matter-only correction.
- New experiments or new baselines:
  rejected because they failed the do-no-harm filter.
- Workshop-mode switch:
  rejected because no verified official workshop target is available.
