# Cycle 01 Review

## Reviewer 1
Score tendency: borderline accept if positioning is fixed.

Strengths:
- The paper has a real empirical question.
- The method is easy to understand and easy to implement.
- The held-out partner evaluation is more interesting than plain self-play return.

Major concerns:
- The novelty is not yet positioned sharply enough against Deep Sets, permutation-invariant critics, and Other-Play.
- The introduction reads as if the paper is apologizing for mixed results instead of arguing for a concrete insight.
- There is no dedicated related-work section, which makes the contribution look less mature than it is.

What would change my score:
- Add a clear positioning section that explains exactly what is new here.
- Reframe the title and abstract around the question the paper answers, not around a vague mismatch slogan.

## Reviewer 2
Score tendency: weak reject in current form.

Strengths:
- The central diagnostic is plausible.
- The results are honest.

Major concerns:
- Same-family cross-play is used as the headline metric, but the paper does not justify why this is the primary endpoint.
- The evaluation protocol is underspecified in the main text. I want to know how many held-out partners each ego policy faces and how the partner pool is filtered.
- The statistical story is underdeveloped. Means and standard errors alone are not enough when the main claim is about instability.

What would change my score:
- State the partner counts and evaluation episodes explicitly.
- Add paired-seed uncertainty reporting and explain why it is the right comparison.

## Reviewer 3
Score tendency: weak accept after revision.

Strengths:
- The mixed result is scientifically useful if framed well.
- The paper avoids overclaiming compared with many workshop submissions.

Major concerns:
- The abstract is not pulling enough weight. It should state the exact empirical pattern more directly.
- The paper is missing a broader-impact discussion even though the checklist appears to expect one.
- The contributions list is close to the right story but still too generic.

What would change my score:
- Rewrite the abstract and contributions around the durable finding.
- Add a short broader-impact paragraph so the checklist and paper match.

## Cycle 01 Takeaway
The paper has workshop value, but the first-pass rejection risk is framing. Reviewers could mistake an honest controlled result for a thin negative result unless the paper explicitly claims the diagnostic-versus-coordination lesson as its main contribution.
