import numpy as np

from picr_ma.envs.observation_slicing import make_contiguous_slicing
from picr_ma.methods.picr_ma import build_picr_views, symmetric_kl, symmetric_kl_with_gradients


def test_sym_kl_zero_same():
    probs = np.array([[0.5, 0.5]], dtype=np.float32)
    assert symmetric_kl(probs, probs) == 0.0


def test_picr_views_preserve_shape():
    slicing = make_contiguous_slicing(obs_dim=6, self_dim=2, teammate_block_dim=2, n_teammates=2)
    obs = np.array([[1, 0, 2, 3, 4, 5]], dtype=np.float32)
    left, right = build_picr_views(obs, slicing, np.random.default_rng(0))
    assert left.shape == obs.shape
    assert right.shape == obs.shape


def test_symmetric_kl_gradients_shape():
    left = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    right = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    loss, grad_left, grad_right = symmetric_kl_with_gradients(left, right)
    assert loss >= 0.0
    assert grad_left.shape == left.shape
    assert grad_right.shape == right.shape
