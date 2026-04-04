import numpy as np

from src.envs.observation_slicing import make_contiguous_slicing
from src.models.actor_critic import build_actor_critic


def test_mlp_policy_outputs_probabilities():
    model = build_actor_critic("mlp", input_dim=6, action_dim=3, seed=0)
    forward = model.forward(np.zeros((2, 6), dtype=np.float32))
    assert forward.probs.shape == (2, 3)
    assert np.allclose(forward.probs.sum(axis=1), 1.0)


def test_deepsets_policy_outputs_probabilities():
    slicing = make_contiguous_slicing(obs_dim=6, self_dim=2, teammate_block_dim=2, n_teammates=2)
    model = build_actor_critic("deepsets", input_dim=6, action_dim=2, seed=0, slicing=slicing)
    forward = model.forward(np.zeros((3, 6), dtype=np.float32))
    assert forward.probs.shape == (3, 2)
    assert np.allclose(forward.probs.sum(axis=1), 1.0)
