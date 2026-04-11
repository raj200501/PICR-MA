# PICR Safe Upgrade Plan

Timestamp: 2026-04-10 17:45:57 EDT

## Chosen candidate improvement

### 1. Front-matter focus correction

What will be changed in candidate space only:

- Tighten the abstract so it foregrounds the main 5-seed, 50k-step MPE cohort.
- Demote matrix-game mention to appendix-only sanity-check status in the abstract and contribution framing.

Why this passed the do-no-harm filter:

- It does not change the method, results, claims, or numbers.
- It strengthens first-impression workshop fit by putting the strongest evidence first.
- It is fully reversible.
- It requires no new computation and no code-path changes.

## Success criteria required before adoption

The candidate wording will be adopted only if all of the following are true:

1. The revised front matter does not add any new scientific claim.
2. The revised wording makes the main-evidence hierarchy clearer:
   the MPE cohort is primary, matrix games are support only.
3. The revised wording does not make the paper sound smaller, more negative, or more defensive than the baseline.
4. The candidate package compiles cleanly in `neurips_candidate_safe/`.
5. A side-by-side comparison shows a real clarity gain, not just different phrasing.

## Improvements intentionally left unchanged

- Title remains unchanged unless the front-matter-only pass is insufficient and a title change is clearly better.
- All tables, figures, and numeric claims remain unchanged.
- No new experiments, no new baselines, and no implementation rewrites will be attempted.
- Workshop mode remains generic because no verified official target is available.
- The protected baseline package in the repo root will not be modified.
