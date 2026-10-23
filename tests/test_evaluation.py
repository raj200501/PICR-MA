import numpy as np

from picr_ma.envs.mpe import make_env
from picr_ma.evaluation import EVAL_SEED_OFFSET, eval_seeds, js_divergence, permutation_sensitivity, play_episodes
from picr_ma.networks import ActorCritic


def test_js_divergence_bounds():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    js = js_divergence(p, q)
    np.testing.assert_allclose(js, [np.log(2), 0.0], atol=1e-9)


def test_permutation_sensitivity_zero_for_deepsets_positive_for_mlp():
    spec = make_env("simple_spread_v3").spec
    obs = np.random.default_rng(0).normal(size=(64, spec.obs_dim)).astype(np.float32)
    deepsets = ActorCritic(spec.obs_dim, spec.n_actions, encoder="deepsets", slicing=spec.slicing)
    assert permutation_sensitivity(deepsets, obs, spec.slicing)["perm_js"] < 1e-10
    mlp = ActorCritic(spec.obs_dim, spec.n_actions)
    mlp.policy_head.weight.data.normal_()
    assert permutation_sensitivity(mlp, obs, spec.slicing)["perm_js"] > 1e-4


def test_eval_is_deterministic_and_uses_reserved_seeds():
    assert min(eval_seeds(10)) >= EVAL_SEED_OFFSET
    spec = make_env("simple_adversary_v3").spec
    team = [ActorCritic(spec.obs_dim, spec.n_actions) for _ in range(spec.n_team)]
    adversary = ActorCritic(spec.adversary_obs_dim, spec.n_actions)
    a = play_episodes(spec.name, team, adversary, eval_seeds(3), sample_seed=5)["returns"]
    b = play_episodes(spec.name, team, adversary, eval_seeds(3), sample_seed=5)["returns"]
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3,)


def test_cross_play_slots_follow_team_list():
    """A team of identical policies must score the same however the list is built."""
    spec = make_env("simple_spread_v3").spec
    policy = ActorCritic(spec.obs_dim, spec.n_actions)
    a = play_episodes(spec.name, [policy] * 3, None, eval_seeds(4), sample_seed=1)["returns"]
    b = play_episodes(spec.name, [policy, policy, policy], None, eval_seeds(4), sample_seed=1)["returns"]
    np.testing.assert_array_equal(a, b)


def test_adversary_pool_is_assigned_round_robin():
    """With a pool, episode i faces adversary i % K; a pool of copies equals a single adversary."""
    spec = make_env("simple_adversary_v3").spec
    team = [ActorCritic(spec.obs_dim, spec.n_actions)] * spec.n_team
    still = ActorCritic(spec.adversary_obs_dim, spec.n_actions)
    mover = ActorCritic(spec.adversary_obs_dim, spec.n_actions)
    for model, action in [(still, 0), (mover, 2)]:  # deterministic: always the same action
        model.policy_head.weight.data.zero_()
        model.policy_head.bias.data.fill_(-50.0)
        model.policy_head.bias.data[action] = 50.0
    seeds = eval_seeds(6)
    mixed = play_episodes(spec.name, team, [still, mover], seeds, sample_seed=3)["returns"]
    only_still = play_episodes(spec.name, team, still, seeds, sample_seed=3)["returns"]
    only_mover = play_episodes(spec.name, team, mover, seeds, sample_seed=3)["returns"]
    np.testing.assert_allclose(mixed[0::2], only_still[0::2])
    np.testing.assert_allclose(mixed[1::2], only_mover[1::2])
