import csv
import json

import numpy as np
import pandas as pd

from picr_ma.analysis.aggregate import compare_to, crossplay_block, load_summaries, method_estimates
from picr_ma.analysis.stats import bootstrap_mean, bootstrap_pairs, difference, permutation_test
from picr_ma.crossplay import evaluate_ego, find_runs
from picr_ma.train import TrainConfig, train


def test_bootstrap_pairs_excludes_self_pairs():
    matrix = np.full((4, 4), 1.0)
    np.fill_diagonal(matrix, 100.0)
    est = bootstrap_pairs(matrix, same_population=True, n_boot=200)
    assert est.mean == 1.0
    assert est.lo == est.hi == 1.0  # resampling a run twice must never pair it with itself


def test_bootstrap_ci_covers_mean_and_difference_detects_shift():
    rng = np.random.default_rng(0)
    a = bootstrap_mean(rng.normal(1.0, 0.1, 10), seed=1)
    b = bootstrap_mean(rng.normal(0.0, 0.1, 10), seed=2)
    assert a.lo < a.mean < a.hi
    diff = difference(a, b)
    assert diff["ci_lo"] > 0
    null = difference(a, bootstrap_mean(rng.normal(1.0, 0.1, 10), seed=3))
    assert null["ci_lo"] < 0 < null["ci_hi"]


def test_permutation_test_detects_cross_play_shift_and_is_calibrated():
    rng = np.random.default_rng(0)
    n = 6
    # Group A coordinates better with itself (high within-A cross-play); B does not.
    matrix = rng.normal(0.0, 0.1, (2 * n, 2 * n))
    matrix[:n, :n] += 1.0
    per_run = {"perm_js": np.r_[np.full(n, 0.1), np.full(n, 0.5)] + rng.normal(0, 0.01, 2 * n)}
    p = permutation_test(matrix, per_run, n_a=n, n_perm=2000)
    assert p["xp"] < 0.01 and p["perm_js"] < 0.01
    # Under the null (exchangeable runs) p-values are roughly uniform.
    null_p = [permutation_test(rng.normal(0, 1, (2 * n, 2 * n)), {}, n_a=n, n_perm=200, seed=s)["xp"] for s in range(60)]
    assert 0.03 < np.mean(np.array(null_p) < 0.1) < 0.25


def test_end_to_end_pipeline(tmp_path):
    """Tiny suite: train -> cross-play -> per-method estimates."""
    env = "simple_spread_v3"
    for method in ["ippo", "picr"]:
        for seed in [0, 1, 2]:
            config = TrainConfig(env=env, method=method, seed=seed, total_steps=200, num_envs=4, eval_episodes=4, log_every=1)
            train(config, tmp_path / "suite" / env / method / f"seed{seed}")
    runs = find_runs(tmp_path / "suite" / env)
    assert len(runs) == 6
    rows = [row for ego in runs for row in evaluate_ego(ego, runs, episodes=4)]
    assert len(rows) == 36
    with (tmp_path / "suite" / env / "crossplay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    xp = pd.read_csv(tmp_path / "suite" / env / "crossplay.csv")
    block = crossplay_block(xp, "ippo", "ippo")
    assert block.shape == (3, 3)
    summaries = load_summaries(tmp_path / "suite")
    estimates = method_estimates(summaries, xp, env, methods=("ippo", "picr"))
    off_diag = block[~np.eye(3, dtype=bool)].mean()
    assert np.isclose(estimates["ippo"]["xp"].mean, off_diag)
    assert np.isclose(estimates["ippo"]["sp"].mean, np.diag(block).mean())
    assert np.isclose(estimates["ippo"]["gap"].mean, np.diag(block).mean() - off_diag)
    comparison = compare_to(estimates, summaries, xp, env, n_perm=50)
    assert set(comparison) == {"picr"}
    assert 0 < comparison["picr"]["xp"]["p"] <= 1
    summary = json.loads((tmp_path / "suite" / env / "picr" / "seed0" / "summary.json").read_text())
    assert summary["picr_lambda"] > 0
