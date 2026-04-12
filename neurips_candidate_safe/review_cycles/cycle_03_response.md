# Cycle 03 Response and Fixes

## Main fixes applied
- Expanded the method section to explain why PICR-MA uses a soft symmetric-KL regularizer instead of exact architectural set pooling.
- Added an explicit sentence that the penalty applies only to exchangeable teammate slices from environment-specific metadata.
- Rewrote the implementation paragraph so it is factual in the main text and leaves the scale critique to the limitations section.
- Reframed the ablation paragraph around the practical takeaway: mild regularization helps most, stronger regularization only improves JS, and teammate dropout is not useful here.
- Clarified in the permutation-sensitivity caption that the subsequent mismatch is not easy to dismiss as a measurement artifact.
- Clarified in the runtime caption that PICR-MA pays training-time cost without enlarging the deployed actor.

## Files affected
- `neurips_candidate_safe/final.tex`

## Acceptance effect
This cycle closes the most likely method-clarity objections without adding risky new claims.
