from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from picr_ma.envs.observation_slicing import ObservationSlicing, make_contiguous_slicing


PAYOFFS = {
    "matrix_coordination": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    "matrix_staghunt": np.array([[1.0, -0.2], [0.8, 0.6]], dtype=np.float32),
}


@dataclass
class MatrixGameConfig:
    name: str
    n_agents: int = 3
    obs_dim: int = 6
    action_dim: int = 2
    max_steps: int = 1

    @property
    def teammate_slicing(self) -> ObservationSlicing:
        return make_contiguous_slicing(
            obs_dim=self.obs_dim,
            self_dim=2,
            teammate_block_dim=2,
            n_teammates=self.n_agents - 1,
            name=f"{self.name}_slicing",
        )


class MatrixGameEnv:
    def __init__(self, name: str, seed: int = 0) -> None:
        if name not in PAYOFFS:
            raise KeyError(f"Unknown matrix game: {name}")
        self.config = MatrixGameConfig(name=name)
        self.payoff = PAYOFFS[name]
        self.rng = np.random.default_rng(seed)
        self.agents = [f"agent_{index}" for index in range(self.config.n_agents)]
        self.step_count = 0
        self.context = 0.0

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.context = float(self.rng.integers(0, 2))
        observations = {}
        for agent_index, agent in enumerate(self.agents):
            self_features = np.array([1.0, self.context], dtype=np.float32)
            teammate_blocks = []
            for teammate_offset in range(1, self.config.n_agents):
                teammate_index = (agent_index + teammate_offset) % self.config.n_agents
                teammate_blocks.append(
                    np.array(
                        [
                            float(teammate_index),
                            float(teammate_offset),
                        ],
                        dtype=np.float32,
                    )
                )
            observations[agent] = np.concatenate([self_features] + teammate_blocks)
        return observations, {}

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        self.step_count += 1
        action_vec = [actions[agent] for agent in self.agents]
        pair_rewards = []
        for left in range(self.config.n_agents):
            for right in range(left + 1, self.config.n_agents):
                pair_rewards.append(self.payoff[action_vec[left], action_vec[right]])
        reward = float(np.mean(pair_rewards))
        success = int(len(set(action_vec)) == 1)

        observations = {
            agent: np.zeros(self.config.obs_dim, dtype=np.float32) for agent in self.agents
        }
        terminated = self.step_count >= self.config.max_steps
        rewards = {agent: reward for agent in self.agents}
        terminateds = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {
            agent: {
                "success": success,
                "team_reward": reward,
            }
            for agent in self.agents
        }
        return observations, rewards, terminateds, truncations, infos

    def observation_space(self, agent: str) -> tuple[int]:
        return (self.config.obs_dim,)

    def action_space(self, agent: str) -> int:
        return self.config.action_dim


def make_matrix_env(name: str, seed: int = 0) -> MatrixGameEnv:
    return MatrixGameEnv(name=name, seed=seed)
