# PICR Verify Before Write

Timestamp: 2026-04-10 17:45:57 EDT

Scope: candidate-paper claim map for the front-matter-only revision trial.

No new scientific claims are planned in this pass. The only candidate wording changes restate evidence that already exists elsewhere in the protected baseline.

| Claim text | Supporting artifact | Exact file path | Support type |
| --- | --- | --- | --- |
| The paper's strongest honest identity is an invariance-coordination mismatch study rather than a universal method-win paper. | Paired seed-level deltas and cross-method summaries show stable diagnostic improvement but mixed coordination effects. | `/Users/rajkashikar/PICR-MA/results/aggregate/paired_picr_vs_ippo.csv`; `/Users/rajkashikar/PICR-MA/results/aggregate/paired_picr_vs_ippo_stats.json`; `/Users/rajkashikar/PICR-MA/results/aggregate/final_metrics_for_paper.json` | Inferential |
| The main evidence is a 5-seed, 50k-step MPE cohort on `simple_spread_v3` and `simple_adversary_v3`. | Generated paper metrics and run summaries for the hardmain cohort. | `/Users/rajkashikar/PICR-MA/results/aggregate/final_metrics_for_paper.json`; `/Users/rajkashikar/PICR-MA/results/aggregate/training_summary.csv` | Direct |
| Matrix coordination and stag-hunt results are retained only as sanity checks rather than the main claim. | Baseline experiment-setup wording and optional appendix figure. | `/Users/rajkashikar/PICR-MA/final.tex`; `/Users/rajkashikar/PICR-MA/latex/figures/optional_matrix_games_summary.pdf` | Direct |
| The headline cross-play statistic is same-family held-out partner cross-play. | Evaluation code averages only `same_method_rows` into `summary["crossplay_return"]`. | `/Users/rajkashikar/PICR-MA/scripts/evaluate.py` | Direct |

If a candidate wording change cannot be justified by one of the artifacts above, it will not be added.
