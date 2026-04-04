# Implementation Credibility

## Decision

- **Retain the current learner for this hardening pass**

## Why this is the decision

Preferred in principle:

- a clean PyTorch IPPO port would be more credible to reviewers

Why I am not choosing that path immediately:

- `torch` is not currently installed in the local environment
- the current repo already has a working end-to-end training/evaluation/manuscript pipeline
- a full PyTorch port would be a high-risk rewrite touching the core learner, checkpointing, evaluation, and possibly analysis compatibility
- before paying that rewrite cost, the more important question is whether PICR-MA clears the scientific bar at all under stronger scale and cleaner evaluation

In other words:

- the first job of this hardening pass is to determine whether the method story survives stricter evidence
- if the method story fails under a hardened NumPy pipeline, a PyTorch port would not rescue the paper

## What this means for the bar

Retaining the current learner is only acceptable if all of the following become true:

- the learner remains stable under materially larger runs
- the baselines are trained fairly under the same budgets
- the cross-play protocol is clean
- the results are strong enough that the implementation choice does not dominate reviewer skepticism
- the paper is framed honestly as a lightweight empirical study, not as a definitive large-scale benchmark

If those conditions do not hold, the final verdict must still be FAIL.

## Credibility risks that remain

1. The learner is custom and NumPy-based rather than a recognizable PyTorch IPPO stack.
2. Reviewers may worry that performance differences are tied to implementation quirks.
3. The current runtime scale is so small that stability and convergence are not yet convincing.
4. Deep Sets comparisons are especially sensitive to fair capacity and optimization.

## Hardening plan implied by this decision

- fix evaluation contamination first
- scale seeds and step counts substantially
- add ablations and overhead analysis
- test whether the learner remains stable under larger budgets
- if the scientific picture is still weak, fail the project rather than hide behind polish

## Current conclusion

- Retaining the learner is a **temporary hardening decision**, not an endorsement that the implementation is already conference-grade.
- The project still needs stronger evidence before this choice can be defended.
