from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationSlicing:
    obs_dim: int
    teammate_blocks: tuple[tuple[int, ...], ...]
    name: str = "manual"

    @property
    def n_teammates(self) -> int:
        return len(self.teammate_blocks)

    @property
    def block_dim(self) -> int:
        return len(self.teammate_blocks[0]) if self.teammate_blocks else 0

    @property
    def self_indices(self) -> tuple[int, ...]:
        taken = {index for block in self.teammate_blocks for index in block}
        return tuple(index for index in range(self.obs_dim) if index not in taken)


def make_contiguous_slicing(
    obs_dim: int,
    self_dim: int,
    teammate_block_dim: int,
    n_teammates: int,
    name: str = "contiguous",
) -> ObservationSlicing:
    teammate_blocks = []
    offset = self_dim
    for _ in range(n_teammates):
        teammate_blocks.append(tuple(range(offset, offset + teammate_block_dim)))
        offset += teammate_block_dim
    return ObservationSlicing(obs_dim=obs_dim, teammate_blocks=tuple(teammate_blocks), name=name)


def split_self_teammates(
    obs: np.ndarray | list[float],
    slicing: ObservationSlicing,
) -> tuple[np.ndarray, np.ndarray]:
    obs_arr = np.asarray(obs, dtype=np.float32)
    self_features = obs_arr[list(slicing.self_indices)]
    teammate_blocks = np.stack(
        [obs_arr[list(block)] for block in slicing.teammate_blocks],
        axis=0,
    )
    return self_features, teammate_blocks


def permute_teammates(
    obs: np.ndarray | list[float],
    slicing: ObservationSlicing,
    permutation: list[int] | tuple[int, ...],
) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32).copy()
    blocks = [obs_arr[list(block)].copy() for block in slicing.teammate_blocks]
    for target_block, source_index in zip(slicing.teammate_blocks, permutation):
        obs_arr[list(target_block)] = blocks[source_index]
    return obs_arr


def zero_teammates(
    obs: np.ndarray | list[float],
    slicing: ObservationSlicing,
    zero_mask: list[bool] | np.ndarray,
) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32).copy()
    for block, drop in zip(slicing.teammate_blocks, zero_mask):
        if drop:
            obs_arr[list(block)] = 0.0
    return obs_arr


def permutation_list(slicing: ObservationSlicing) -> list[tuple[int, ...]]:
    if slicing.n_teammates <= 1:
        return [tuple(range(slicing.n_teammates))]
    if slicing.n_teammates == 2:
        return [(0, 1), (1, 0)]
    return [tuple(range(slicing.n_teammates)), tuple(reversed(range(slicing.n_teammates)))]
