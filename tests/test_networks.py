import numpy as np
import torch

from picr_ma.envs.slicing import TeammateSlicing, permute_teammates
from picr_ma.networks import ActorCritic, load_policy, save_policy

SLICING = TeammateSlicing(obs_dim=10, teammate_blocks=((2, 3, 4), (5, 6, 7)))


def _obs(n=64):
    return np.random.default_rng(0).normal(size=(n, 10)).astype(np.float32)


def test_deepsets_is_exactly_invariant_and_mlp_is_not():
    obs = _obs()
    swapped = permute_teammates(obs, SLICING, np.array([1, 0]))
    deepsets = ActorCritic(10, 5, encoder="deepsets", slicing=SLICING)
    np.testing.assert_allclose(deepsets.action_probs(obs), deepsets.action_probs(swapped), atol=1e-6)
    mlp = ActorCritic(10, 5, encoder="mlp")
    mlp.policy_head.weight.data.normal_()  # default init is near-uniform; make outputs informative
    assert np.abs(mlp.action_probs(obs) - mlp.action_probs(swapped)).max() > 1e-3


def test_act_returns_consistent_log_probs():
    model = ActorCritic(10, 5)
    actions, log_probs, values = model.act(_obs(8), torch.Generator().manual_seed(0))
    probs = model.action_probs(_obs(8))
    np.testing.assert_allclose(np.log(probs[np.arange(8), actions]), log_probs, rtol=1e-5)
    assert values.shape == (8,)


def test_save_load_roundtrip(tmp_path):
    model = ActorCritic(10, 5, encoder="deepsets", slicing=SLICING)
    save_policy(model, tmp_path / "p.pt", SLICING)
    loaded = load_policy(tmp_path / "p.pt")
    np.testing.assert_allclose(model.action_probs(_obs()), loaded.action_probs(_obs()), atol=1e-7)
