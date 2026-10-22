#!/usr/bin/env python
"""Reference returns on simple_spread_v3 for calibrating learned policies.

    python scripts/reference_returns.py

Scores three non-learned controllers on the same reserved evaluation seeds as
the learned policies:

* random     - uniform random discrete actions
* no-op      - every agent stays still
* assignment - a scripted controller that, at the first step, assigns agents
               to landmarks by minimum total distance (exhaustive over the 3!
               assignments) and then steers each agent to its landmark with a
               velocity-damped bang-bang rule. It uses global state, so it is a
               strong reference point, not an upper bound.

Writes results/reference_returns.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from picr_ma.envs.mpe import make_env
from picr_ma.evaluation import eval_seeds

ROOT = Path(__file__).resolve().parents[1]


def _steer(rel: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Discrete action that pushes along the dominant axis of (target - 0.5 * velocity)."""
    target = rel - 0.5 * vel
    horizontal = np.abs(target[..., 0]) > np.abs(target[..., 1])
    action = np.where(horizontal, np.where(target[..., 0] > 0, 2, 1), np.where(target[..., 1] > 0, 4, 3))
    return np.where(np.abs(target).max(axis=-1) < 0.02, 0, action)


def run(controller: str, episodes: int, seed: int = 0) -> np.ndarray:
    env = make_env("simple_spread_v3")
    rng = np.random.default_rng(seed)
    env.reset(eval_seeds(episodes))
    perms = np.array(list(itertools.permutations(range(3))))
    totals = np.stack([np.linalg.norm(env.pos - env.landmarks[:, p], axis=-1).sum(-1) for p in perms], axis=1)
    assignment = perms[totals.argmin(axis=1)]  # (B, agent) -> landmark index
    returns = np.zeros(episodes)
    done = False
    while not done:
        if controller == "random":
            actions = rng.integers(0, 5, (episodes, 3))
        elif controller == "noop":
            actions = np.zeros((episodes, 3), dtype=np.int64)
        else:
            target = env.landmarks[np.arange(episodes)[:, None], assignment]
            actions = _steer(target - env.pos, env.vel)
        _, rewards, _, done = env.step(actions)
        returns += rewards.mean(axis=1)
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    episodes = args.episodes
    out = {}
    for controller in ["random", "noop", "assignment"]:
        returns = run(controller, episodes)
        out[controller] = {"mean": float(returns.mean()), "se": float(returns.std() / np.sqrt(episodes))}
        print(f"{controller:>10}: {returns.mean():7.2f} +- {returns.std() / np.sqrt(episodes):.2f}")
    path = args.results_dir / "reference_returns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": "simple_spread_v3", "episodes": episodes, **out}, indent=2))


if __name__ == "__main__":
    main()
