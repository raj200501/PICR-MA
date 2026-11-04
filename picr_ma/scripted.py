"""Scripted simple_spread_v3 policies, used as reference points and as positive controls.

The positive controls check that the evaluation would detect the effects the
study looks for, if they existed:

* ``GreedyAssignment`` follows a *convention*: agents claim landmarks greedily
  in west-to-east (or east-to-west) order of their positions. A team that
  shares the convention covers every landmark; mixing the two conventions
  makes agents collide on the same landmark. It reads teammates only through
  their positions, so it is invariant to teammate order.
* ``FirstTeammateLandmark`` is *order-dependent*: it heads for the landmark
  nearest to whichever teammate is listed in a given slot.

Policies expose the ``act`` interface of :class:`picr_ma.networks.ActorCritic`.
"""

from __future__ import annotations

import numpy as np
import torch

# simple_spread_v3 observation layout for N=3 (see picr_ma/envs/mpe.py).
VEL, LANDMARKS, TEAMMATE_POS = slice(0, 2), slice(4, 10), ((10, 11), (12, 13))


def steer(rel: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Discrete action pushing along the dominant axis of (target - 0.5 * velocity); 0 when close."""
    target = rel - 0.5 * vel
    horizontal = np.abs(target[..., 0]) > np.abs(target[..., 1])
    action = np.where(horizontal, np.where(target[..., 0] > 0, 2, 1), np.where(target[..., 1] > 0, 4, 3))
    return np.where(np.abs(target).max(axis=-1) < 0.02, 0, action)


class ScriptedPolicy:
    """Deterministic target-seeking policy with an optional epsilon of uniformly random actions."""

    n_actions = 5

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def target(self, obs: np.ndarray) -> np.ndarray:  # (B, 2) target position relative to the agent
        raise NotImplementedError

    def act(self, obs: np.ndarray, generator: torch.Generator | None = None, greedy: bool = False):
        obs = np.asarray(obs, dtype=np.float64)
        actions = steer(self.target(obs), obs[:, VEL])
        if self.epsilon > 0 and not greedy:
            explore = torch.rand(len(obs), generator=generator).numpy() < self.epsilon
            random = torch.randint(0, self.n_actions, (len(obs),), generator=generator).numpy()
            actions = np.where(explore, random, actions)
        zeros = np.zeros(len(obs), dtype=np.float32)
        return actions.astype(np.int64), zeros, zeros

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        probs = np.full((len(obs), self.n_actions), self.epsilon / self.n_actions)
        probs[np.arange(len(obs)), steer(self.target(obs), obs[:, VEL])] += 1 - self.epsilon
        return probs


def _landmarks(obs: np.ndarray) -> np.ndarray:
    return obs[:, LANDMARKS].reshape(len(obs), -1, 2)


class GreedyAssignment(ScriptedPolicy):
    """Agents claim their nearest free landmark one at a time, ordered west-to-east (or east-to-west).

    Every agent sees every position, so all agents following the same convention
    compute the same assignment and cover all landmarks.
    """

    def __init__(self, direction: str = "west", epsilon: float = 0.0) -> None:
        super().__init__(epsilon)
        if direction not in ("west", "east"):
            raise ValueError(direction)
        self.direction = direction

    def target(self, obs: np.ndarray) -> np.ndarray:
        landmarks = _landmarks(obs)
        agents = np.stack([np.zeros((len(obs), 2))] + [obs[:, list(block)] for block in TEAMMATE_POS], axis=1)  # self first
        order = np.argsort(agents[..., 0] * (1 if self.direction == "west" else -1), axis=1, kind="stable")
        targets = np.empty((len(obs), 2))
        for b in range(len(obs)):
            free = list(range(landmarks.shape[1]))
            for agent in order[b]:
                distances = [np.linalg.norm(landmarks[b, lm] - agents[b, agent]) for lm in free]
                claimed = free.pop(int(np.argmin(distances)))
                if agent == 0:
                    targets[b] = landmarks[b, claimed]
                    break
        return targets


class FirstTeammateLandmark(ScriptedPolicy):
    """Heads for the landmark nearest to the teammate listed in ``slot``: an order-dependent rule."""

    def __init__(self, slot: int = 0, epsilon: float = 0.0) -> None:
        super().__init__(epsilon)
        self.slot = slot

    def target(self, obs: np.ndarray) -> np.ndarray:
        landmarks = _landmarks(obs)
        teammate = obs[:, list(TEAMMATE_POS[self.slot])]
        nearest = np.linalg.norm(landmarks - teammate[:, None], axis=-1).argmin(axis=1)
        return landmarks[np.arange(len(obs)), nearest]
