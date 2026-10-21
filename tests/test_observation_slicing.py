import numpy as np

from picr_ma.envs.observation_slicing import ObservationSlicing, permute_teammates, split_self_teammates, zero_teammates


def test_split_and_permute_manual_blocks():
    slicing = ObservationSlicing(obs_dim=8, teammate_blocks=((2, 3), (6, 7)))
    obs = np.array([9, 8, 1, 2, 7, 6, 3, 4], dtype=np.float32)
    self_features, teammates = split_self_teammates(obs, slicing)
    assert self_features.tolist() == [9, 8, 7, 6]
    assert teammates.tolist() == [[1, 2], [3, 4]]
    swapped = permute_teammates(obs, slicing, (1, 0))
    assert swapped.tolist() == [9, 8, 3, 4, 7, 6, 1, 2]
    dropped = zero_teammates(obs, slicing, [True, False])
    assert dropped.tolist() == [9, 8, 0, 0, 7, 6, 3, 4]
