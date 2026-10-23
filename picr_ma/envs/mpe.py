"""Batched NumPy implementation of two PettingZoo MPE scenarios.

``simple_spread_v3`` and ``simple_adversary_v3`` (N=3, discrete actions,
25-step episodes) are re-implemented so that B environments step together as
array operations. Physics, rewards, observations and seeded resets follow the
reference implementation in the ``mpe2`` package (the maintained successor of
``pettingzoo.mpe``); ``tests/test_envs.py`` checks that trajectories match the
reference step for step under the same seeds and actions.

Observation layouts (N=3):

simple_spread_v3, 18 dims per agent:
    [self_vel(2), self_pos(2), landmark_rel(3x2), other_rel(2x2), other_comm(2x2)]
    Agents are silent, so the comm entries are always zero.
    teammate k = other_rel[k] + other_comm[k]  -> indices (10,11,14,15), (12,13,16,17)

simple_adversary_v3, 14 dims per good agent:
    [goal_rel(2), landmark_rel(3x2), adversary_rel(2), good_teammate_rel(2x2)]
    teammate k -> indices (10,11), (12,13)
    adversary, 12 dims: [landmark_rel(3x2), good_agent_rel(3x2)]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from picr_ma.envs.slicing import TeammateSlicing

ENV_NAMES = ("simple_spread_v3", "simple_adversary_v3")

DT = 0.1
DAMPING = 0.25
SENSITIVITY = 5.0
CONTACT_FORCE = 1e2
CONTACT_MARGIN = 1e-3
# Discrete action -> unit force direction: no-op, -x, +x, -y, +y.
ACTION_DIRECTIONS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class EnvSpec:
    name: str
    obs_dim: int
    n_actions: int
    n_team: int
    slicing: TeammateSlicing
    n_adversaries: int = 0
    adversary_obs_dim: int = 0
    horizon: int = 25


def _spread_slicing(n_agents: int) -> TeammateSlicing:
    n_others = n_agents - 1
    pos_start = 4 + 2 * n_agents
    comm_start = pos_start + 2 * n_others
    blocks = tuple(
        (pos_start + 2 * k, pos_start + 2 * k + 1, comm_start + 2 * k, comm_start + 2 * k + 1) for k in range(n_others)
    )
    return TeammateSlicing(obs_dim=comm_start + 2 * n_others, teammate_blocks=blocks)


def _adversary_slicing(n_good: int) -> TeammateSlicing:
    start = 2 + 2 * n_good + 2
    blocks = tuple((start + 2 * k, start + 2 * k + 1) for k in range(n_good - 1))
    return TeammateSlicing(obs_dim=start + 2 * (n_good - 1), teammate_blocks=blocks)


def _others(n: int) -> np.ndarray:
    """(n, n-1) index array: row i lists every agent except i, in order."""
    return np.array([[j for j in range(n) if j != i] for i in range(n)], dtype=np.int64)


class BatchedMPE:
    """B synchronized copies of one scenario. All episodes share the same horizon."""

    def __init__(self, name: str, n_agents: int = 3, horizon: int = 25) -> None:
        if name not in ENV_NAMES:
            raise KeyError(f"Unknown environment {name!r}; expected one of {ENV_NAMES}")
        self.name = name
        self.n = n_agents
        self.horizon = horizon
        if name == "simple_spread_v3":
            self.spec = EnvSpec(name, 6 * n_agents, 5, n_agents, _spread_slicing(n_agents), horizon=horizon)
            self.n_world_agents = n_agents
        else:
            self.spec = EnvSpec(
                name,
                4 * n_agents + 2,
                5,
                n_agents,
                _adversary_slicing(n_agents),
                n_adversaries=1,
                adversary_obs_dim=4 * n_agents,
                horizon=horizon,
            )
            self.n_world_agents = n_agents + 1  # world order: [adversary_0, agent_0, ..., agent_{N-1}]
        self.t = 0

    # ------------------------------------------------------------------ reset
    def reset(self, seeds: Sequence[int]) -> dict[str, np.ndarray]:
        b = len(seeds)
        self.pos = np.empty((b, self.n_world_agents, 2))
        self.landmarks = np.empty((b, self.n, 2))
        self.goal = np.zeros(b, dtype=np.int64)
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))  # == gymnasium.utils.seeding.np_random(seed)
            if self.name == "simple_adversary_v3":
                self.goal[i] = rng.integers(self.n)
            for a in range(self.n_world_agents):
                self.pos[i, a] = rng.uniform(-1, 1, 2)
            for lm in range(self.n):
                self.landmarks[i, lm] = rng.uniform(-1, 1, 2)
        self.vel = np.zeros_like(self.pos)
        self.t = 0
        return self._observe()

    # ------------------------------------------------------------------- step
    def step(self, team_actions: np.ndarray, adversary_actions: np.ndarray | None = None):
        """Advance every environment one step.

        Returns (observations, team_rewards (B, N), adversary_rewards (B, n_adv),
        done) where ``done`` is True once the horizon is reached.
        """
        team_actions = np.asarray(team_actions, dtype=np.int64)
        if self.name == "simple_adversary_v3":
            adversary_actions = np.zeros((len(team_actions), 1), np.int64) if adversary_actions is None else adversary_actions
            actions = np.concatenate([np.asarray(adversary_actions, dtype=np.int64), team_actions], axis=1)
        else:
            actions = team_actions
        force = SENSITIVITY * ACTION_DIRECTIONS[actions]
        if self.name == "simple_spread_v3":
            force = force + self._contact_forces(self.pos, size=0.15)
        self.pos = self.pos + self.vel * DT
        self.vel = self.vel * (1 - DAMPING) + force * DT
        self.t += 1
        team_rewards, adversary_rewards = self._rewards()
        return self._observe(), team_rewards, adversary_rewards, self.t >= self.horizon

    @staticmethod
    def _contact_forces(pos: np.ndarray, size: float) -> np.ndarray:
        delta = pos[:, :, None, :] - pos[:, None, :, :]  # (B, n, n, 2): pos_a - pos_b
        dist = np.sqrt(np.sum(np.square(delta), axis=-1))
        n = pos.shape[1]
        off_diag = ~np.eye(n, dtype=bool)
        safe = np.where(off_diag, dist, 1.0)
        penetration = np.logaddexp(0, -(safe - 2 * size) / CONTACT_MARGIN) * CONTACT_MARGIN
        pair_force = CONTACT_FORCE * delta / safe[..., None] * penetration[..., None]
        return np.sum(np.where(off_diag[None, :, :, None], pair_force, 0.0), axis=2)

    def _rewards(self) -> tuple[np.ndarray, np.ndarray]:
        b = len(self.pos)
        if self.name == "simple_spread_v3":
            to_lm = np.linalg.norm(self.pos[:, :, None] - self.landmarks[:, None], axis=-1)  # (B, agent, lm)
            global_reward = -to_lm.min(axis=1).sum(axis=1)  # (B,)
            dist = np.linalg.norm(self.pos[:, :, None] - self.pos[:, None], axis=-1)
            collisions = ((dist < 0.3) & ~np.eye(self.n, dtype=bool)).sum(axis=2)  # (B, agent)
            # local_ratio = 0.5: reward = 0.5 * global + 0.5 * (-collisions)
            team = 0.5 * global_reward[:, None] + 0.5 * (-collisions.astype(np.float64))
            return team, np.zeros((b, 0))
        goal = self.landmarks[np.arange(b), self.goal]  # (B, 2)
        adv_dist = np.linalg.norm(self.pos[:, 0] - goal, axis=-1)
        good_dist = np.linalg.norm(self.pos[:, 1:] - goal[:, None], axis=-1).min(axis=1)
        team = np.repeat((adv_dist - good_dist)[:, None], self.n, axis=1)
        return team, -adv_dist[:, None]

    # -------------------------------------------------------------- observe
    def _observe(self) -> dict[str, np.ndarray]:
        others = _others(self.n_world_agents)
        rel_agents = self.pos[:, others] - self.pos[:, :, None]  # (B, agent, n-1, 2)
        rel_lms = self.landmarks[:, None] - self.pos[:, :, None]  # (B, agent, lm, 2)
        b = len(self.pos)
        if self.name == "simple_spread_v3":
            comm = np.zeros((b, self.n, 2 * (self.n - 1)))
            team = np.concatenate(
                [self.vel, self.pos, rel_lms.reshape(b, self.n, -1), rel_agents.reshape(b, self.n, -1), comm], axis=-1
            )
            return {"team": team.astype(np.float32)}
        goal = self.landmarks[np.arange(b), self.goal]
        goal_rel = goal[:, None] - self.pos  # (B, world_agent, 2)
        world_obs = np.concatenate(
            [goal_rel, rel_lms.reshape(b, self.n_world_agents, -1), rel_agents.reshape(b, self.n_world_agents, -1)], axis=-1
        )
        adversary = np.concatenate([rel_lms[:, :1].reshape(b, 1, -1), rel_agents[:, :1].reshape(b, 1, -1)], axis=-1)
        return {"team": world_obs[:, 1:].astype(np.float32), "adversary": adversary.astype(np.float32)}


def make_env(name: str, horizon: int = 25) -> BatchedMPE:
    return BatchedMPE(name, horizon=horizon)


def env_spec(name: str) -> EnvSpec:
    return BatchedMPE(name).spec
