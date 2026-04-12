# Verify Before Write

Only claims with a concrete support path were added or sharpened in the paper.

| Paper claim | Supporting artifact | Exact file path | Support type |
| --- | --- | --- | --- |
| The main-table cross-play number is same-family held-out partner cross-play rather than an all-partners average. | Evaluation code writes `summary["crossplay_return"]` from `same_method_rows` only. | `scripts/evaluate.py` | Direct |
| On `simple_spread_v3`, PICR-MA lowers permutation sensitivity in all 5 matched IPPO/PICR seed pairs and loses same-family cross-play in all 5. | Paired seed-level deltas. | `results/aggregate/paired_picr_vs_ippo.csv` | Direct |
| On `simple_spread_v3`, the mean same-family cross-play delta is `-1.557` with a 95% bootstrap interval of `[-3.013, -0.651]`. | Bootstrapped paired summary. | `results/aggregate/paired_picr_vs_ippo_stats.json` | Direct |
| On `simple_adversary_v3`, PICR-MA improves same-family cross-play in 3 of 5 matched pairs, but the 95% bootstrap interval for the mean delta still spans zero. | Paired seed-level counts and bootstrap summary. | `results/aggregate/paired_picr_vs_ippo.csv`; `results/aggregate/paired_picr_vs_ippo_stats.json` | Direct |
| IPPO, PICR-MA, and Other-Play share the same inference-time MLP capacity, while Deep Sets changes capacity. | Generated implementation summary. | `results/aggregate/implementation_summary.csv` | Direct |
| PICR-MA runtime is about `1.62x` IPPO on `simple_spread_v3` and `1.33x` on `simple_adversary_v3`. | Generated implementation summary and runtime table. | `results/aggregate/implementation_summary.csv`; `latex/tables/runtime_results.tex` | Direct |
| PICR-MA's positive adversary result is concentrated in same-family partners rather than a uniform cross-family gain. | Same-family versus cross-family appendix table. | `latex/tables/crossplay_results.tex` | Inferential |

If a claim could not be traced to a generated file or code path in the repo, it was not added.
