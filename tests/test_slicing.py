import itertools

import numpy as np
import pytest
import torch

from picr_ma.envs.slicing import TeammateSlicing, permute_teammates, random_permutations, split_self_teammates

SLICING = TeammateSlicing(obs_dim=8, teammate_blocks=((2, 3), (6, 7)))
OBS = np.array([[9, 8, 1, 2, 7, 6, 3, 4]], dtype=np.float32)


def test_permute_swaps_blocks_and_keeps_self_features():
    swapped = permute_teammates(OBS, SLICING, np.array([1, 0]))
    assert swapped.tolist() == [[9, 8, 3, 4, 7, 6, 1, 2]]
    assert permute_teammates(OBS, SLICING, np.array([0, 1])).tolist() == OBS.tolist()


def test_numpy_and_torch_agree_and_rows_use_their_own_permutation():
    slicing = TeammateSlicing(obs_dim=7, teammate_blocks=((1, 2), (3, 4), (5, 6)))
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(6, 7)).astype(np.float32)
    perms = np.array(list(itertools.permutations(range(3))))
    np_out = permute_teammates(obs, slicing, perms)
    torch_out = permute_teammates(torch.as_tensor(obs), slicing, perms).numpy()
    np.testing.assert_array_equal(np_out, torch_out)
    for row, perm in enumerate(perms):
        for slot, source in enumerate(perm):
            dst, src = slicing.teammate_blocks[slot], slicing.teammate_blocks[source]
            np.testing.assert_array_equal(np_out[row, list(dst)], obs[row, list(src)])
        np.testing.assert_array_equal(np_out[row, [0]], obs[row, [0]])


def test_random_permutations_are_uniform():
    perms = random_permutations(30_000, 3, np.random.default_rng(0))
    assert all(sorted(p) == [0, 1, 2] for p in perms[:100])
    _, counts = np.unique(perms, axis=0, return_counts=True)
    assert len(counts) == 6
    assert counts.min() / counts.max() > 0.9


def test_split_self_teammates():
    self_feats, teammates = split_self_teammates(torch.as_tensor(OBS), SLICING)
    assert self_feats.tolist() == [[9, 8, 7, 6]]
    assert teammates.tolist() == [[[1, 2], [3, 4]]]


def test_invalid_slicing_rejected():
    with pytest.raises(ValueError):
        TeammateSlicing(obs_dim=4, teammate_blocks=((0, 1), (1, 2)))
    with pytest.raises(ValueError):
        TeammateSlicing(obs_dim=4, teammate_blocks=((0, 1), (2,)))
