import csv
import gzip
import json

import numpy as np
import pytest

from picr_ma.envs.mpe import make_env
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


@pytest.mark.parametrize("method", METHODS)
def test_each_method_trains_with_its_own_mechanism(tmp_path, monkeypatch, method):
    """PICR's penalty is active only for PICR; order_aug permutes inputs at every step;
    order_rand holds one permutation per agent for the whole episode; others see raw inputs."""
    import picr_ma.train as train_module

    calls = []
    original = train_module._policy_input

    def recording(method, obs, slicing, rng, episode_perms=None):
        out = original(method, obs, slicing, rng, episode_perms)
        calls.append((obs.copy(), out.copy()))
        return out

    monkeypatch.setattr(train_module, "_policy_input", recording)
    config = TrainConfig(env="simple_spread_v3", method=method, seed=0, total_steps=200, num_envs=4, eval_episodes=4, log_every=1)
    train(config, tmp_path)
    with gzip.open(tmp_path / "metrics.csv.gz", "rt") as handle:
        picr_losses = [float(row["picr_loss"]) for row in csv.DictReader(handle)]
    assert (max(picr_losses) > 0) == (method == "picr")

    slicing = make_env("simple_spread_v3").spec.slicing
    teammate, own = list(np.ravel(slicing.teammate_blocks)), list(slicing.self_indices)
    raw, seen = np.stack([c[0] for c in calls]), np.stack([c[1] for c in calls])  # (steps, rows, D)
    np.testing.assert_array_equal(raw[..., own], seen[..., own])
    if method in ("ippo", "picr", "deepsets"):
        np.testing.assert_array_equal(raw, seen)
        return
    swapped = (seen[..., teammate] != raw[..., teammate]).any(axis=-1)  # (steps, rows): rows whose order changed
    assert 0.2 < swapped.mean() < 0.8  # two teammates: half the permutations swap them
    episode = slice(0, 25)  # the first episode's 25 steps
    if method == "order_rand":
        assert (swapped[episode] == swapped[0]).all()  # fixed within the episode
    else:
        assert not (swapped[episode] == swapped[0]).all()
