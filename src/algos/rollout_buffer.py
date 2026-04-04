from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBatch:
    observations: list[np.ndarray] = field(default_factory=list)
    raw_observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def append(
        self,
        observation: np.ndarray,
        raw_observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.raw_observations.append(np.asarray(raw_observation, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def size(self) -> int:
        return len(self.actions)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "raw_observations": np.asarray(self.raw_observations, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=bool),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.raw_observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
