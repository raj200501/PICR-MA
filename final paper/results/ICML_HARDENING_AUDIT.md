# ICML Hardening Audit

## Branch status

- Working branch: `icml_hardening_pass`
- Main status: not approved
- Current outcome: `FAIL` pending in `results/ICML_GO_NO_GO.md`

## 1. What is currently implemented?

- Shared-parameter decentralized on-policy learner with `ippo`, `picr_ma`, `deepsets`, and `otherplay` variants.
- Explicit teammate-block slicing for `simple_spread_v3`, `simple_adversary_v3` with `N=3` good agents, and 3-agent matrix games.
- PICR-MA symmetric KL consistency loss over two independently permuted teammate-order views.
- Held-out partner cross-play evaluation from saved checkpoints.
- Permutation-sensitivity evaluation via Jensen-Shannon divergence.
- Aggregate CSV/JSON exports, PDF plots, LaTeX table fragments, and a populated `main.tex`.

Hardening changes in this pass:

- Fixed cross-play partner-pool leakage: evaluation now scopes partners to the intended `run_tag` cohort instead of the whole `results/` tree.
- Hardened the Other-Play baseline: teammate-order randomization is now episode-consistent instead of resampling every step.
- Added summary bookkeeping for `picr_lambda`, teammate dropout, eval episodes, and parameter counts in new runs.
- Updated the manuscript text so it reflects the 50k/5-seed evidence instead of the earlier 1k-step toy runs.

## 2. What experiments have actually been run?

- Re-evaluated all tracked `stage1` runs with cohort-scoped partner pools.
- `hardpilot20k`: 24 runs
  - environments: `simple_spread_v3`, `simple_adversary_v3`
  - methods: `ippo`, `picr_ma`, `deepsets`, `otherplay`
  - seeds: 3
  - steps: 20k
- `lambda20k`: 18 runs
  - environments: `simple_spread_v3`, `simple_adversary_v3`
  - method: `picr_ma`
  - seeds: 3
  - steps: 20k
  - lambdas: `0.02`, `0.05`, `0.10`
- `drop20k_p010`: 6 runs
  - environments: `simple_spread_v3`, `simple_adversary_v3`
  - method: `picr_ma`
  - seeds: 3
  - steps: 20k
  - lambda: `0.02`
  - teammate dropout: `0.10`
- `hardmain50k`: 40 runs
  - environments: `simple_spread_v3`, `simple_adversary_v3`
  - methods: `ippo`, `picr_ma`, `deepsets`, `otherplay`
  - seeds: 5
  - steps: 50k
  - PICR-MA lambda: `0.02`

Recorded training runtime from the hardening cohorts:

- `hardpilot20k`: `777.489` s
- `lambda20k`: `1017.618` s
- `drop20k_p010`: `100.225` s
- `hardmain50k`: `1655.531` s

## 3. What are the current seeds, step counts, and runtime budgets?

- Main reported MPE cohort: 5 seeds, 50k env steps, 16 in-team eval episodes, 16 cross-play episodes.
- Main profile: `PROFILE_LEAN_SAFE`
- Current disk use:
  - `results/`: about `31M`
  - free disk at end of hardening pass: about `4.8 GiB`
- Main-cohort mean runtime per run:
  - `simple_spread_v3`: IPPO `31.004` s, PICR-MA `50.235` s, Deep Sets `62.529` s, Other-Play `32.329` s
  - `simple_adversary_v3`: IPPO `33.504` s, PICR-MA `44.575` s, Deep Sets `44.885` s, Other-Play `32.045` s

## 4. Which results currently support the paper's main claim?

Supportive evidence is limited to `simple_adversary_v3`:

- PICR-MA improves held-out same-family cross-play over IPPO:
  - PICR-MA: `20.159 ± 2.043`
  - IPPO: `17.429 ± 3.770`
- PICR-MA also reduces permutation sensitivity relative to IPPO on the same environment:
  - PICR-MA: `0.0053 ± 0.0025`
  - IPPO: `0.0102 ± 0.0066`
- PICR-MA decisively outperforms Deep Sets on adversary cross-play:
  - PICR-MA: `20.159 ± 2.043`
  - Deep Sets: `11.356 ± 3.244`

The lambda sweep also shows that the default regularization strength was not viable:

- At 20k steps, `lambda=0.02` was the best PICR-MA setting for held-out cross-play.
- Larger lambda values lowered JS more aggressively but hurt `simple_spread_v3`.

## 5. Which results currently undermine or weaken the claim?

The main weakness is `simple_spread_v3`:

- PICR-MA is the worst of the four methods on held-out same-family cross-play:
  - PICR-MA: `-27.206 ± 0.730`
  - IPPO: `-25.649 ± 0.271`
  - Deep Sets: `-25.550 ± 0.489`
  - Other-Play: `-25.533 ± 0.566`

Additional weakening evidence:

- The strongest invariance settings are not the best coordination settings.
  - At 20k, `lambda=0.10` lowers spread JS to `0.0091` but worsens spread cross-play to `-34.673`.
- The best cross-play lambda (`0.02`) does not improve JS over IPPO in the 20k pilot.
  - `simple_spread_v3`: PICR `0.0247 ± 0.0065` vs IPPO `0.0229 ± 0.0002`
  - `simple_adversary_v3`: PICR `0.0156 ± 0.0061` vs IPPO `0.0098 ± 0.0042`
- Teammate dropout is harmful.
  - `simple_spread_v3`: `-37.486 ± 2.504` cross-play with dropout vs `-31.673 ± 0.520` without
  - `simple_adversary_v3`: `16.897 ± 2.086` with dropout vs `25.743 ± 1.344` without
- Across individual runs in the 50k main cohort, the correlation between lower JS and better cross-play is weakly positive rather than negative:
  - `simple_spread_v3`: `corr(JS, crossplay) = 0.106`
  - `simple_adversary_v3`: `corr(JS, crossplay) = 0.143`

## 6. Is PICR-MA clearly better than IPPO?

No.

- Better on `simple_adversary_v3`
- Worse on `simple_spread_v3`
- No consistent overall advantage across the main MPE tasks

## 7. Is PICR-MA clearly better than Deep Sets?

No.

- Better on `simple_adversary_v3`
- Worse on `simple_spread_v3`
- Deep Sets remains the strongest invariance baseline by construction (`JS = 0`)

## 8. Is PICR-MA clearly better than Other-Play?

No.

- Other-Play is slightly better on adversary cross-play (`20.934 ± 3.429` vs `20.159 ± 2.043`)
- Other-Play is clearly better on spread cross-play (`-25.533 ± 0.566` vs `-27.206 ± 0.730`)
- Other-Play also runs closer to IPPO cost than PICR-MA

## 9. Does the paper currently rely more on aesthetics than evidence?

Less than before, but the evidence is still not strong enough for a serious ICML-style pass.

- Before hardening: yes, the paper looked stronger than the evidence.
- After hardening: the text now matches the evidence, but the evidence itself is mixed and the appendix package remains incomplete.

## 10. What are the top 5 reviewer objections right now?

1. The headline claim is mixed: PICR-MA helps on `simple_adversary_v3` but loses on `simple_spread_v3`, so it is not a robust new default.
2. The mechanistic invariance story is weak: lower JS does not reliably predict better cross-play, and the best 20k cross-play setting is not the most invariant one.
3. The scale is still modest by conference standards: the main MPE cohort is 50k steps with 25-cycle episodes, and `simple_spread_v3` curves are still improving late in training.
4. Baseline pressure remains high: Other-Play is at least as strong on adversary and clearly stronger on spread; Deep Sets is competitive on spread and dominates the invariance diagnostic.
5. Implementation credibility is still limited by the lightweight NumPy learner and the lack of a more standard PyTorch PPO stack.
