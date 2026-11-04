import numpy as np
import pytest

from picr_ma.envs.mpe import make_env
from picr_ma.envs.slicing import relabeling_for, teammate_views, view_patterns
from picr_ma.evaluation import EVAL_SEED_OFFSET, eval_seeds, js_divergence, permutation_sensitivity, play_episodes
from picr_ma.networks import ActorCritic
from picr_ma.scripted import FirstTeammateLandmark, GreedyAssignment

SPREAD = make_env("simple_spread_v3").spec


def _constant(obs_dim: int, action: int) -> ActorCritic:
    """A policy that always takes ``action``."""
    model = ActorCritic(obs_dim, 5)
    model.policy_head.weight.data.zero_()
    model.policy_head.bias.data.fill_(-50.0)
    model.policy_head.bias.data[action] = 50.0
    return model


def _order_sensitive_mlp(seed: int = 0) -> ActorCritic:
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    model = ActorCritic(SPREAD.obs_dim, SPREAD.n_actions)
    model.policy_head.weight.data.normal_(0, 3.0)
    return model


def test_js_divergence_bounds():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(js_divergence(p, q), [np.log(2), 0.0], atol=1e-9)


def test_permutation_sensitivity_zero_for_deepsets_positive_for_mlp():
    obs = np.random.default_rng(0).normal(size=(64, SPREAD.obs_dim)).astype(np.float32)
    deepsets = ActorCritic(SPREAD.obs_dim, SPREAD.n_actions, encoder="deepsets", slicing=SPREAD.slicing)
    assert permutation_sensitivity(deepsets, obs, SPREAD.slicing)["perm_js"] < 1e-10
    assert permutation_sensitivity(_order_sensitive_mlp(), obs, SPREAD.slicing)["perm_js"] > 1e-4


def test_evaluation_seed_ranges_are_disjoint_from_training_and_each_other():
    test, validation = set(eval_seeds(1000, "test")), set(eval_seeds(1000, "validation"))
    assert min(test | validation) >= EVAL_SEED_OFFSET  # training draws seeds below the offset
    assert not test & validation


def test_returns_are_the_mean_team_reward_summed_over_time():
    policy = _order_sensitive_mlp()
    seeds = eval_seeds(4)
    env = make_env("simple_spread_v3")
    obs, expected, done = env.reset(seeds), np.zeros(4), False
    while not done:
        actions = np.stack([policy.act(obs["team"][:, i], greedy=True)[0] for i in range(3)], axis=1)
        obs, rewards, _, done = env.step(actions)
        expected += rewards.mean(axis=1)
    np.testing.assert_allclose(play_episodes(SPREAD.name, [policy] * 3, None, seeds, greedy=True)["returns"], expected)


def test_eval_is_deterministic():
    spec = make_env("simple_adversary_v3").spec
    team = [ActorCritic(spec.obs_dim, spec.n_actions) for _ in range(spec.n_team)]
    adversary = ActorCritic(spec.adversary_obs_dim, spec.n_actions)
    a = play_episodes(spec.name, team, adversary, eval_seeds(3), sample_seed=5)["returns"]
    b = play_episodes(spec.name, team, adversary, eval_seeds(3), sample_seed=5)["returns"]
    np.testing.assert_array_equal(a, b)


def test_adversary_pool_is_assigned_to_episode_pairs():
    """Episodes 2k and 2k+1 face adversary k % K, so even and odd halves see the same opponents."""
    spec = make_env("simple_adversary_v3").spec
    team = [ActorCritic(spec.obs_dim, spec.n_actions)] * spec.n_team
    still, mover = _constant(spec.adversary_obs_dim, 0), _constant(spec.adversary_obs_dim, 2)
    seeds = eval_seeds(8)
    mixed = play_episodes(spec.name, team, [still, mover], seeds, sample_seed=3)["returns"]
    only_still = play_episodes(spec.name, team, still, seeds, sample_seed=3)["returns"]
    only_mover = play_episodes(spec.name, team, mover, seeds, sample_seed=3)["returns"]
    faces_still = np.array([0, 0, 1, 1, 0, 0, 1, 1]) == 0
    np.testing.assert_allclose(mixed[faces_still], only_still[faces_still])
    np.testing.assert_allclose(mixed[~faces_still], only_mover[~faces_still])


def test_cross_play_puts_the_ego_in_each_slot():
    from picr_ma.crossplay import returns_by_slot

    ego, partner = _constant(SPREAD.obs_dim, 0), _constant(SPREAD.obs_dim, 2)
    seeds = eval_seeds(16)
    per_slot = returns_by_slot(SPREAD.name, ego, partner, None, seeds)
    for slot, returns in enumerate(per_slot):
        team = [partner] * 3
        team[slot] = ego
        np.testing.assert_array_equal(returns, play_episodes(SPREAD.name, team, None, seeds)["returns"])
    partner_only = play_episodes(SPREAD.name, [partner] * 3, None, seeds)["returns"]
    assert not np.allclose(np.mean(per_slot, axis=0), partner_only)


def test_view_patterns_for_three_agents():
    patterns = view_patterns(3, SPREAD.slicing)
    kinds = [p["kind"] for p in patterns]
    assert (kinds.count("identity"), kinds.count("relabeling"), kinds.count("genuine")) == (1, 5, 2)
    genuine = {tuple(teammate_views(3, p["slot_perms"])) for p in patterns if p["kind"] == "genuine"}
    assert genuine == {((1, 2), (2, 0), (0, 1)), ((2, 1), (0, 2), (1, 0))}  # the two cyclic conventions


def _greedy_rollout(policy, env_name, seeds, relabel=None):
    env = make_env(env_name)
    obs = env.reset(seeds)
    if relabel is not None:  # agent i of the original episode gets label relabel[i]
        pos = env.pos.copy()
        for agent, label in enumerate(relabel):
            env.pos[:, label] = pos[:, agent]
        obs = env._observe()
    returns, done = np.zeros(len(seeds)), False
    while not done:
        actions = np.stack([policy.act(obs["team"][:, i], greedy=True)[0] for i in range(3)], axis=1)
        obs, rewards, _, done = env.step(actions)
        returns += rewards.mean(axis=1)
    return returns


def test_relabeling_patterns_are_exactly_relabelings_and_genuine_patterns_are_not():
    """For a deterministic policy, a relabeling pattern reproduces the canonical rollout on the
    relabeled start state, episode by episode. A genuine pattern changes an order-dependent policy."""
    policy, seeds = _order_sensitive_mlp(), eval_seeds(32)
    identity = play_episodes(SPREAD.name, [policy] * 3, None, seeds, greedy=True)["returns"]
    for pattern in view_patterns(3, SPREAD.slicing):
        shifted = play_episodes(SPREAD.name, [policy] * 3, None, seeds, greedy=True, slot_permutations=pattern["slot_perms"])
        shifted = shifted["returns"]
        relabel = relabeling_for(teammate_views(3, pattern["slot_perms"]))
        if pattern["kind"] == "genuine":
            assert relabel is None
            assert not np.allclose(shifted, identity)
        else:
            np.testing.assert_allclose(shifted, _greedy_rollout(policy, SPREAD.name, seeds, relabel), atol=1e-9)


def test_order_shift_detects_an_order_dependent_scripted_policy_and_not_an_invariant_one():
    from picr_ma.crossplay import order_shift_returns

    seeds = eval_seeds(240)
    for policy, dependent in [(FirstTeammateLandmark(0, 0.05), True), (GreedyAssignment("west", 0.05), False)]:
        rows = order_shift_returns(SPREAD.name, policy, None, seeds)
        base = next(r["return_mean"] for r in rows if r["kind"] == "identity")
        change = {k: np.mean([r["return_mean"] for r in rows if r["kind"] == k]) - base for k in ("relabeling", "genuine")}
        if dependent:
            assert abs(change["genuine"]) > 1.0 > 2 * abs(change["relabeling"])
        else:
            assert change["genuine"] == change["relabeling"] == 0.0


def test_resampling_replaces_teammate_blocks_with_another_episodes():
    seen = []

    class Recorder(ActorCritic):
        def act(self, obs, generator=None, greedy=False):
            seen.append(obs.copy())
            return super().act(obs, generator, greedy)

    seeds = eval_seeds(5)
    policy = Recorder(SPREAD.obs_dim, SPREAD.n_actions)
    play_episodes(SPREAD.name, [policy] * 3, None, seeds, sample_seed=0, resample_teammates=True)
    resampled_first_step = np.stack(seen[:3], axis=1)  # (episodes, agents, D) at t=0
    original = make_env(SPREAD.name).reset(seeds)["team"]
    teammate, own = list(np.ravel(SPREAD.slicing.teammate_blocks)), list(SPREAD.slicing.self_indices)
    np.testing.assert_array_equal(resampled_first_step[..., own], original[..., own])
    np.testing.assert_array_equal(resampled_first_step[..., teammate], np.roll(original, 1, axis=0)[..., teammate])


def test_order_shift_matches_cross_play_diagonal_and_leaves_invariant_policies_unchanged(tmp_path):
    from picr_ma.crossplay import evaluate_ego, evaluate_order_shift, find_runs
    from picr_ma.train import TrainConfig, train

    for method in ["ippo", "deepsets"]:
        config = TrainConfig(env=SPREAD.name, method=method, seed=0, total_steps=200, num_envs=4, eval_episodes=4, log_every=1)
        train(config, tmp_path / SPREAD.name / method / "seed0")
    for run in find_runs(tmp_path / SPREAD.name):
        rows = evaluate_order_shift(run, episodes=8)
        assert [r["kind"] for r in rows].count("genuine") == 2 and rows[-1]["kind"] == "resampled"
        identity = next(r["return_mean"] for r in rows if r["kind"] == "identity")
        assert np.isclose(identity, evaluate_ego(run, [run], episodes=8)[0]["return_mean"], rtol=1e-12)
        if run.method == "deepsets":
            assert all(r["return_mean"] == identity for r in rows if r["kind"] != "resampled")
