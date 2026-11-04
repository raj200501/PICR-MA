# Results

Everything the paper reports is computed from the files here by
`scripts/analyze.py` (`make analyze`), with no training or evaluation.
`configs/experiments.yaml` defines every run; `tests/test_committed_results.py`
checks that the runs here are exactly those and that the current code
reproduces their training logs.

## Layout

```
lambda/            choosing the PICR weight: seeds 100-104, validation episodes
  selection.json   the selected weight per environment and the score of each candidate
  <env>/picr_lam<λ>/seed<k>/, <env>/ippo/seed<k>/   (IPPO is the λ = 0 reference)
main/              the main comparison: seeds 0-9, test episodes
  <env>/<method>/seed<k>/
long/              twice the training budget (10M steps): simple_spread_v3, seeds 20-24
tuning/ippo_sweep.jsonl   the IPPO hyperparameter sweep (docs/tuning.md)
reference_returns.json    random, no-op and scripted-assignment controllers
positive_control.json     scripted controllers through the same evaluation (instrument checks)
env_speed.json            batched environment vs. mpe2, steps per second
overhead.json             cost of one PPO update and one forward pass, per method
analysis.json             every estimate, comparison and statistic in the paper
main_results.md           the main table in Markdown
```

`<env>` is `simple_spread_v3` or `simple_adversary_v3`; `<method>` is `ippo`,
`picr`, `deepsets`, `order_aug` or `order_rand`.

## One training run (`<suite>/<env>/<method>/seed<k>/`)

| File | Contents |
|---|---|
| `config.json` | the full `TrainConfig`, including PPO hyperparameters |
| `metrics.csv.gz` | every 5th PPO update: environment steps, training return, PPO losses, approximate KL, clip fraction, PICR penalty, permutation JS and flip rate on the training batch |
| `policy.pt` | the final team policy (`picr_ma.networks.load_policy`) |
| `adversary.pt` | the co-trained adversary (`simple_adversary_v3` only) |
| `summary.json` | training time, parameter count, and self-play return and permutation sensitivity of the final policy on 500 test-range episodes. In `simple_adversary_v3` this self-play is against the run's own adversary, a training diagnostic; the paper's numbers use the shared adversary pool |

## Evaluation outputs (`<suite>/<env>/`)

`crossplay.csv` has one row per ordered pair of runs (ego, partner), including
self-play (ego = partner):

| Column | Meaning |
|---|---|
| `ego_method`, `ego_seed`, `partner_method`, `partner_seed` | the pairing; the ego controls one agent, the partner the other two |
| `return_mean`, `return_std` | team return per episode, averaged over the ego's three seats |
| `return_mean_half1`, `return_mean_half2` | the same over even and odd episodes (for split-half reliability) |
| `return_mean_slot0` … `slot2` | with the ego in each seat; slot 0 vs. the last slot gives the slot contrast |
| `episodes`, `split`, `adversary_pool_size` | evaluation episodes, which split, size of the shared adversary pool |

`order_shift.csv` has one row per run and teammate-order condition: `kind` is
`identity`, `relabeling` (a null control), `genuine` (an order no relabeling
produces) or `resampled` (teammate features from another episode); `pattern`
gives each agent's teammate ordering.

`crossplay.provenance.json` and `order_shift.provenance.json` record the git
commit, whether the code had uncommitted changes, and the Python, platform,
CPU, NumPy and PyTorch versions that produced the CSV. The other JSON outputs
carry the same record under `provenance`.
