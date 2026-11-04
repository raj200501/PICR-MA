"""Teammate-block structure of flat observation vectors.

An observation is a flat vector in which each teammate occupies a fixed set of
indices (a "block"). Teammates are exchangeable, so the order of the blocks is
arbitrary. This module describes the blocks and permutes them in batch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TeammateSlicing:
    obs_dim: int
    teammate_blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        widths = {len(block) for block in self.teammate_blocks}
        if len(widths) > 1:
            raise ValueError(f"Teammate blocks must share a width, got {sorted(widths)}")
        flat = [i for block in self.teammate_blocks for i in block]
        if len(flat) != len(set(flat)):
            raise ValueError("Teammate blocks overlap")
        if flat and (min(flat) < 0 or max(flat) >= self.obs_dim):
            raise ValueError("Teammate block index out of range")

    @property
    def n_teammates(self) -> int:
        return len(self.teammate_blocks)

    @property
    def block_dim(self) -> int:
        return len(self.teammate_blocks[0]) if self.teammate_blocks else 0

    @property
    def self_indices(self) -> tuple[int, ...]:
        taken = {i for block in self.teammate_blocks for i in block}
        return tuple(i for i in range(self.obs_dim) if i not in taken)

    def all_permutations(self, include_identity: bool = True) -> list[tuple[int, ...]]:
        perms = list(itertools.permutations(range(self.n_teammates)))
        if not include_identity:
            perms = [p for p in perms if p != tuple(range(self.n_teammates))]
        return perms

    def gather_index(self, perms: np.ndarray) -> np.ndarray:
        """Index array that applies a teammate permutation to each row.

        ``perms`` has shape (batch, n_teammates); row ``b`` places teammate
        ``perms[b, k]`` into slot ``k``. The result has shape (batch, obs_dim) and
        is used as ``obs[b, index[b]]``.
        """
        perms = np.asarray(perms, dtype=np.int64)
        blocks = np.asarray(self.teammate_blocks, dtype=np.int64)  # (n_teammates, block_dim)
        index = np.broadcast_to(np.arange(self.obs_dim), (len(perms), self.obs_dim)).copy()
        if self.n_teammates:
            index[:, blocks.reshape(-1)] = blocks[perms].reshape(len(perms), -1)
        return index


def permute_teammates(obs, slicing: TeammateSlicing, perms):
    """Permute teammate blocks of a batch of observations (NumPy or torch)."""
    perms = np.asarray(perms)
    if perms.ndim == 1:
        perms = np.broadcast_to(perms, (len(obs), slicing.n_teammates))
    index = slicing.gather_index(perms)
    if isinstance(obs, torch.Tensor):
        return torch.gather(obs, 1, torch.as_tensor(index, device=obs.device))
    return np.take_along_axis(np.asarray(obs), index, axis=1)


def random_permutations(n: int, n_teammates: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` independent uniformly random permutations of ``range(n_teammates)``."""
    return np.argsort(rng.random((n, n_teammates)), axis=1)


def split_self_teammates(obs: torch.Tensor, slicing: TeammateSlicing) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (batch, obs_dim) into self features and (batch, n_teammates, block_dim)."""
    self_idx = torch.as_tensor(slicing.self_indices, device=obs.device)
    team_idx = torch.as_tensor(slicing.teammate_blocks, device=obs.device)
    return obs[:, self_idx], obs[:, team_idx]


def teammate_views(n_agents: int, slot_perms) -> list[tuple[int, ...]]:
    """Which teammate each agent sees in each slot, given one slot permutation per agent.

    Agent ``i`` normally lists its teammates in index order; its permutation
    ``slot_perms[i]`` places canonical teammate ``p[k]`` in slot ``k``.
    """
    views = []
    for agent, perm in enumerate(slot_perms):
        canonical = [j for j in range(n_agents) if j != agent]
        views.append(tuple(canonical[k] for k in perm))
    return views


def relabeling_for(views: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    """A relabeling of the agents that produces exactly these views, or None if there is none.

    Every agent lists its teammates in label order, so a pattern of views is
    induced by a relabeling exactly when one total order of the agents is
    consistent with every agent's list; agent ``i``'s new label is its rank in
    that order. With exchangeable agents and a shared policy, such a pattern
    cannot change expected returns: it is the same game with the agents renamed.
    """
    n = len(views)
    for order in itertools.permutations(range(n)):
        rank = {agent: r for r, agent in enumerate(order)}
        if all(all(rank[a] < rank[b] for a, b in zip(view, view[1:])) for view in views):
            return tuple(rank[agent] for agent in range(n))
    return None


def induced_by_relabeling(views: list[tuple[int, ...]]) -> bool:
    return relabeling_for(views) is not None


def view_patterns(n_agents: int, slicing: TeammateSlicing) -> list[dict]:
    """Every combination of per-agent teammate-slot permutations, labelled by kind.

    ``identity``: all agents see the canonical order. ``relabeling``: a
    non-identity pattern that a renaming of the agents also produces (a null
    control). ``genuine``: a pattern no renaming produces; for three agents these
    are the two cyclic conventions in which agent ``i`` lists ``i+1, i+2`` (or
    the reverse) modulo 3.
    """
    patterns = []
    for slot_perms in itertools.product(slicing.all_permutations(), repeat=n_agents):
        identity = all(p == tuple(range(slicing.n_teammates)) for p in slot_perms)
        relabel = induced_by_relabeling(teammate_views(n_agents, slot_perms))
        kind = "identity" if identity else ("relabeling" if relabel else "genuine")
        patterns.append({"slot_perms": slot_perms, "kind": kind})
    return patterns
