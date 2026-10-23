import gzip
import json

import numpy as np
import pytest

from picr_ma.networks import load_policy
from picr_ma.ppo import PPOConfig
from picr_ma.train import METHODS, TrainConfig, train


@pytest.mark.parametrize("env", ["simple_spread_v3", "simple_adversary_v3"])
@pytest.mark.parametrize("method", METHODS)
def test_train_writes_artifacts(tmp_path, env, method):
    config = TrainConfig(
        env=env, method=method, seed=0, total_steps=200, num_envs=4, eval_episodes=4, log_every=1, ppo=PPOConfig(epochs=1)
    )
    summary = train(config, tmp_path)
    for name in ["config.json", "metrics.csv.gz", "summary.json", "policy.pt"]:
        assert (tmp_path / name).exists()
    assert (tmp_path / "adversary.pt").exists() == (env == "simple_adversary_v3")
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert summary["env_steps"] == 200
    assert (summary["picr_lambda"] > 0) == (method == "picr")
    if method == "deepsets":
        assert summary["perm_js"] < 1e-10
    policy = load_policy(tmp_path / "policy.pt")
    assert np.isfinite(policy.action_probs(np.zeros((1, policy.config["obs_dim"]), dtype=np.float32))).all()


def test_same_seed_is_reproducible(tmp_path):
    config = TrainConfig(env="simple_spread_v3", method="picr", seed=1, total_steps=200, num_envs=4, eval_episodes=4, log_every=1)
    a = train(config, tmp_path / "a")
    b = train(config, tmp_path / "b")
    for key in ["self_play_return", "perm_js"]:
        assert a[key] == b[key]
    assert gzip.decompress((tmp_path / "a" / "metrics.csv.gz").read_bytes()) == gzip.decompress(
        (tmp_path / "b" / "metrics.csv.gz").read_bytes()
    )


def test_runs_have_independent_random_streams():
    """Different methods (or PICR weights) with the same seed must not share initial weights or episodes."""
    from picr_ma.train import run_seed_sequence

    def state(method, lam=0.1, seed=0):
        return tuple(
            run_seed_sequence(TrainConfig(env="simple_spread_v3", method=method, seed=seed, picr_lambda=lam)).generate_state(4)
        )

    assert state("ippo") == state("ippo")
    assert state("ippo") != state("picr")
    assert state("picr", 0.1) != state("picr", 0.3)
    assert state("ippo", 0.1) == state("ippo", 0.3)  # lambda is unused by IPPO
    assert state("ippo", seed=0) != state("ippo", seed=1)


def test_finished_run_with_a_different_config_is_not_reused(tmp_path):
    import pytest

    from picr_ma.experiments import Run

    config = TrainConfig(env="simple_spread_v3", method="ippo", seed=0, total_steps=100, num_envs=4, eval_episodes=4)
    train(config, tmp_path)
    assert Run("s", config, tmp_path).done
    changed = TrainConfig(env="simple_spread_v3", method="ippo", seed=0, total_steps=200, num_envs=4, eval_episodes=4)
    with pytest.raises(RuntimeError, match="different config"):
        _ = Run("s", changed, tmp_path).done
