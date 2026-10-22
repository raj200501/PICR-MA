# Hyperparameter tuning

All four methods share one PPO configuration. It was chosen by tuning **IPPO
only**, on tuning seed 1000 (never used for λ selection or final evaluation),
so the baseline is at least as well tuned as the methods compared against it.
PICR's single extra hyperparameter, λ, is selected separately on seeds 100–104
(`scripts/select_lambda.py`).

Selection criterion: self-play return on `simple_spread_v3` (500 episodes on
the reserved evaluation seeds). The chosen configuration was then checked on
`simple_adversary_v3`. Reference points on the same seeds
(`scripts/reference_returns.py`): uniform random −26.95, no-op −24.78, a
scripted controller with global state that assigns agents to landmarks −9.21.

Chosen: lr 1e-3, 10 epochs, 8 minibatches, 128 episodes per update, entropy
coefficient 0.01, 5M environment steps (training return was still improving
slowly between 4M and 6M steps).

Raw results: `results/tuning/ippo_sweep.jsonl` (regenerate this table with
`python scripts/tuning_table.py`). Each row is a single seed, so differences
of about 1–2 return points are within seed-to-seed noise.

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
