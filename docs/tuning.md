# Hyperparameter tuning

All methods share one PPO configuration. It was chosen by tuning **IPPO
only**, on tuning seed 1000 (a seed used for nothing else), so the baseline is
at least as well tuned as the methods compared against it. PICR's one extra
hyperparameter, λ, is selected separately on seeds 100–104
(`scripts/select_lambda.py`).

**Criterion.** Self-play return on `simple_spread_v3`, over the first 500
test-split evaluation episodes (environment seeds 1,000,000–1,000,499). These
include the 240 test episodes on which the final comparison is scored. Only
IPPO self-play from the tuning seed's runs was used, but the choice of
configuration was made on those episodes. Reference points on the same
episodes (`scripts/reference_returns.py`): uniform random −26.95, no-op −24.78,
a scripted controller with global state that assigns agents to landmarks −9.21.

**Chosen:** lr 1e-3, 10 epochs, 8 minibatches, 128 episodes per update, entropy
coefficient 0.01, 5M environment steps.

**How much the sweep can tell apart.** Each configuration was run with one
seed. With this configuration, IPPO's seed-to-seed SD of self-play return on
`simple_spread_v3` is 1.83 (seeds 0–9 of the main suite, at 5M steps), so two
single-seed configurations differ with an SD of about 2.6. The sweep therefore
only rules out configurations that are clearly worse (by more than about 5
points); the top six rows below cannot be told apart. The chosen
configuration's tuning run (−11.96 at 3M steps) was also a favourable seed: the
same configuration's mean training return at 3M over seeds 0–9 is −14.34
(SD 1.78).

**Check on `simple_adversary_v3`.** The configuration tuned on
`simple_spread_v3` scored 10.42, against 10.89 for the earlier default (lr
1e-3, 5 epochs, 4 minibatches, 64 episodes per update), against each run's own
co-trained adversary. The seed SD there is about 0.25, so a single-seed
difference has an SD of about 0.35: the tuned configuration may be slightly
worse on this environment. One configuration was kept for both environments.

**Step budget.** The 6M-step row is the chosen configuration trained longer on
the tuning seed (−9.54, against −11.96 at 3M); its training curve was not kept.
5M steps was a compute compromise. Every `simple_spread_v3` run in the main
suite was still improving at 5M steps (the paper's appendix gives the slopes),
so methods are compared at a fixed budget rather than at convergence; the
`long` suite repeats the `simple_spread_v3` comparison at 10M steps.

Raw results: `results/tuning/ippo_sweep.jsonl` (regenerate this table with
`python scripts/tuning_table.py`; rerun the sweep with `make tuning`).
Permutation JS is measured as in the main experiments.

| Env | Steps | lr | Epochs | Minibatches | Envs / update | Entropy coef. | Self-play return | Perm. JS |
|---|---|---|---|---|---|---|---|---|
| simple_spread_v3 | 3M | 0.001 | 10 | 8 | 128 | 0.01 | -11.96 | 0.134 |
| simple_spread_v3 | 3M | 0.001 | 15 | 8 | 128 | 0.01 | -13.21 | 0.217 |
| simple_spread_v3 | 3M | 0.002 | 5 | 4 | 64 | 0.01 | -14.34 | 0.127 |
| simple_spread_v3 | 3M | 0.002 | 10 | 8 | 128 | 0.01 | -14.58 | 0.240 |
| simple_spread_v3 | 3M | 0.001 | 10 | 16 | 256 | 0.01 | -15.05 | 0.107 |
| simple_spread_v3 | 3M | 0.0007 | 15 | 4 | 64 | 0.01 | -15.17 | 0.172 |
| simple_spread_v3 | 3M | 0.001 | 10 | 4 | 64 | 0 | -15.96 | 0.161 |
| simple_spread_v3 | 3M | 0.002 | 10 | 4 | 64 | 0.01 | -16.38 | 0.257 |
| simple_spread_v3 | 3M | 0.001 | 10 | 4 | 64 | 0.01 | -17.14 | 0.149 |
| simple_spread_v3 | 3M | 0.001 | 10 | 4 | 32 | 0.01 | -17.72 | 0.238 |
| simple_spread_v3 | 3M | 0.001 | 5 | 4 | 64 | 0.01 | -17.91 | 0.242 |
| simple_spread_v3 | 3M | 0.0003 | 10 | 4 | 64 | 0.01 | -18.23 | 0.103 |
| simple_spread_v3 | 3M | 0.0003 | 5 | 4 | 64 | 0.01 | -18.68 | 0.184 |
| simple_spread_v3 | 6M | 0.001 | 10 | 8 | 128 | 0.01 | -9.54 | 0.132 |
| simple_adversary_v3 | 3M | 0.001 | 5 | 4 | 64 | 0.01 | 10.89 | 0.022 |
| simple_adversary_v3 | 3M | 0.001 | 10 | 8 | 128 | 0.01 | 10.42 | 0.138 |
