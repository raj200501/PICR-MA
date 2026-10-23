"""The batched environments must reproduce the reference mpe2 scenarios exactly."""

import numpy as np
import pytest

from picr_ma.envs.mpe import make_env

pytest.importorskip("mpe2")


def _reference(name):
    from mpe2 import simple_adversary_v3, simple_spread_v3

    if name == "simple_spread_v3":
        return simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    return simple_adversary_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)


@pytest.mark.parametrize("name", ["simple_spread_v3", "simple_adversary_v3"])
def test_trajectories_match_reference(name):
    seeds = [0, 7, 123, 99_999]
    rng = np.random.default_rng(0)
    batched = make_env(name)
    spec = batched.spec
    obs = batched.reset(seeds)
    refs = [_reference(name) for _ in seeds]
    ref_obs = [ref.reset(seed=s)[0] for ref, s in zip(refs, seeds)]
    team_names = [f"agent_{i}" for i in range(spec.n_team)]
    for t in range(25):
        for i, ro in enumerate(ref_obs):
            np.testing.assert_allclose(obs["team"][i], np.stack([ro[a] for a in team_names]), atol=1e-5, err_msg=f"t={t}")
            if spec.n_adversaries:
                np.testing.assert_allclose(obs["adversary"][i, 0], ro["adversary_0"], atol=1e-5)
        team_actions = rng.integers(0, 5, size=(len(seeds), spec.n_team))
        adv_actions = rng.integers(0, 5, size=(len(seeds), spec.n_adversaries))
        obs, team_rewards, adv_rewards, done = batched.step(team_actions, adv_actions)
        for i, ref in enumerate(refs):
            actions = {a: int(x) for a, x in zip(team_names, team_actions[i])}
            if spec.n_adversaries:
                actions["adversary_0"] = int(adv_actions[i, 0])
            ref_obs[i], ref_rewards, _, truncations, _ = ref.step(actions)
            np.testing.assert_allclose(team_rewards[i], [ref_rewards[a] for a in team_names], atol=1e-6)
            if spec.n_adversaries:
                np.testing.assert_allclose(adv_rewards[i, 0], ref_rewards["adversary_0"], atol=1e-6)
            assert done == all(truncations.values())
    assert done


def test_collisions_are_exercised():
    """Agents started on top of each other must be pushed apart by contact forces."""
    env = make_env("simple_spread_v3")
    env.reset([0])
    env.pos[0] = [[0.0, 0.0], [0.05, 0.0], [1.0, 1.0]]
    _, rewards, _, _ = env.step(np.zeros((1, 3), dtype=np.int64))
    assert env.vel[0, 0, 0] < 0 < env.vel[0, 1, 0]
    assert rewards[0, 2] > rewards[0, 0]  # agents 0 and 1 pay the collision penalty


@pytest.mark.parametrize("name", ["simple_spread_v3", "simple_adversary_v3"])
def test_teammate_blocks_hold_teammate_features(name):
    env = make_env(name)
    obs = env.reset([5])["team"][0]
    team_pos = env.pos[0, -env.spec.n_team :]
    for i in range(env.spec.n_team):
        others = [j for j in range(env.spec.n_team) if j != i]
        for block, j in zip(env.spec.slicing.teammate_blocks, others):
            np.testing.assert_allclose(obs[i, list(block[:2])], team_pos[j] - team_pos[i], atol=1e-6)


def test_specs():
    spread = make_env("simple_spread_v3").spec
    assert (spread.obs_dim, spread.n_actions, spread.n_team, spread.n_adversaries) == (18, 5, 3, 0)
    adversary = make_env("simple_adversary_v3").spec
    assert (adversary.obs_dim, adversary.n_team, adversary.n_adversaries, adversary.adversary_obs_dim) == (14, 3, 1, 12)


def test_clustered_variant_spawns_agents_together_and_keeps_spread_dynamics():
    clustered, standard = make_env("spread_clustered"), make_env("simple_spread_v3")
    assert clustered.spec.name == "spread_clustered"
    assert clustered.spec.obs_dim == standard.spec.obs_dim
    assert clustered.spec.slicing == standard.spec.slicing
    clustered.reset(list(range(200)))
    spread = np.linalg.norm(clustered.pos - clustered.pos.mean(axis=1, keepdims=True), axis=-1)
    assert spread.max() < 0.3  # every agent starts within a few cm of the cluster centre
    # After reset, dynamics and rewards are those of simple_spread.
    standard.reset(list(range(200)))
    standard.pos, standard.landmarks, standard.vel = clustered.pos.copy(), clustered.landmarks.copy(), clustered.vel.copy()
    actions = np.random.default_rng(0).integers(0, 5, (200, 3))
    obs_c, rew_c, _, _ = clustered.step(actions)
    obs_s, rew_s, _, _ = standard.step(actions)
    np.testing.assert_allclose(rew_c, rew_s)
    np.testing.assert_allclose(obs_c["team"], obs_s["team"])
