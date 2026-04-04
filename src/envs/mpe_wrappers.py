from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.observation_slicing import ObservationSlicing


try:
    from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3
except Exception:  # pragma: no cover - optional dependency path
    simple_adversary_v3 = None
    simple_spread_v3 = None


@dataclass(frozen=True)
class EnvSpec:
    env_name: str
    action_dim: int
    obs_dim: int
    cooperative_agents: tuple[str, ...]
    teammate_slicing: ObservationSlicing
    max_cycles: int
    adversary_agents: tuple[str, ...] = ()


def _spread_slicing() -> ObservationSlicing:
    return ObservationSlicing(
        obs_dim=18,
        teammate_blocks=((10, 11, 14, 15), (12, 13, 16, 17)),
        name="simple_spread_v3_manual",
    )


def _adversary_slicing(n_good_agents: int) -> ObservationSlicing:
    landmark_dims = 2 * n_good_agents
    # good-agent observation: [goal(2), landmarks(2*N), adversary(2), good teammates...]
    base = 2 + landmark_dims + 2
    teammate_blocks = []
    for teammate_index in range(n_good_agents - 1):
        start = base + 2 * teammate_index
        teammate_blocks.append((start, start + 1))
    return ObservationSlicing(
        obs_dim=2 + landmark_dims + 2 * n_good_agents,
        teammate_blocks=tuple(teammate_blocks),
        name=f"simple_adversary_v3_N{n_good_agents}_manual",
    )


def make_mpe_env(
    env_name: str,
    seed: int,
    max_cycles: int,
):
    if env_name == "simple_spread_v3":
        if simple_spread_v3 is None:
            raise RuntimeError("PettingZoo MPE simple_spread_v3 is unavailable.")
        env = simple_spread_v3.parallel_env(
            N=3,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
        observations, _ = env.reset(seed=seed)
        spec = EnvSpec(
            env_name=env_name,
            action_dim=int(env.action_space(env.agents[0]).n),
            obs_dim=int(observations[env.agents[0]].shape[0]),
            cooperative_agents=tuple(env.agents),
            teammate_slicing=_spread_slicing(),
            max_cycles=max_cycles,
        )
        return env, spec

    if env_name == "simple_adversary_v3":
        if simple_adversary_v3 is None:
            raise RuntimeError("PettingZoo MPE simple_adversary_v3 is unavailable.")
        env = simple_adversary_v3.parallel_env(
            N=3,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
        observations, _ = env.reset(seed=seed)
        good_agents = tuple(agent for agent in env.agents if agent.startswith("agent_"))
        adversary_agents = tuple(agent for agent in env.agents if agent.startswith("adversary_"))
        spec = EnvSpec(
            env_name=env_name,
            action_dim=int(env.action_space(env.agents[0]).n),
            obs_dim=int(observations[good_agents[0]].shape[0]),
            cooperative_agents=good_agents,
            adversary_agents=adversary_agents,
            teammate_slicing=_adversary_slicing(len(good_agents)),
            max_cycles=max_cycles,
        )
        return env, spec

    raise KeyError(f"Unsupported MPE env: {env_name}")


def simple_spread_collisions(env) -> int:
    world = env.unwrapped.world
    collisions = 0
    for left in range(len(world.agents)):
        for right in range(left + 1, len(world.agents)):
            agent_a = world.agents[left]
            agent_b = world.agents[right]
            delta = agent_a.state.p_pos - agent_b.state.p_pos
            distance = float(np.sqrt(np.sum(np.square(delta))))
            if distance < (agent_a.size + agent_b.size):
                collisions += 1
    return collisions


def simple_adversary_goal_margin(env) -> float:
    world = env.unwrapped.world
    goal = world.agents[0].goal_a
    good_agents = [agent for agent in world.agents if not agent.adversary]
    adversary = next(agent for agent in world.agents if agent.adversary)
    good_distance = min(
        float(np.linalg.norm(agent.state.p_pos - goal.state.p_pos))
        for agent in good_agents
    )
    adversary_distance = float(np.linalg.norm(adversary.state.p_pos - goal.state.p_pos))
    return adversary_distance - good_distance
