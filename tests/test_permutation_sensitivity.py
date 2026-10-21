import numpy as np

from picr_ma.envs.observation_slicing import make_contiguous_slicing
from picr_ma.eval.permutation_sensitivity import sensitivity


def test_sensitivity_nonnegative():
    slicing = make_contiguous_slicing(obs_dim=6, self_dim=2, teammate_block_dim=2, n_teammates=2)
    obs = np.array([1, 0, 1, 2, 3, 4], dtype=np.float32)
    policy = lambda batch: np.tile(np.array([[0.6, 0.4]], dtype=np.float32), (len(batch), 1))
    score, values = sensitivity(policy, obs, slicing)
    assert score >= 0.0
    assert len(values) >= 1
