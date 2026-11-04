#!/usr/bin/env python
"""Environment throughput: the batched NumPy MPE vs. the reference mpe2 implementation.

    python scripts/benchmark_env.py

Steps both implementations with uniformly random actions for complete 25-step
episodes of simple_spread_v3 (128 episodes per batch for the batched version,
one environment at a time for mpe2) and reports environment steps per second.
Writes results/env_speed.json. Requires the optional mpe2 dependency.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from picr_ma.envs.mpe import make_env
from picr_ma.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def batched_steps_per_second(episodes: int, batch: int) -> float:
    env, rng = make_env("simple_spread_v3"), np.random.default_rng(0)
    start, steps = time.perf_counter(), 0
    for k in range(episodes // batch):
        env.reset(list(range(k * batch, (k + 1) * batch)))
        done = False
        while not done:
            _, _, _, done = env.step(rng.integers(0, 5, (batch, 3)))
            steps += batch
    return steps / (time.perf_counter() - start)


def reference_steps_per_second(episodes: int) -> float:
    from mpe2 import simple_spread_v3

    env, rng = (
        simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False),
        np.random.default_rng(0),
    )
    start, steps = time.perf_counter(), 0
    for seed in range(episodes):
        env.reset(seed=seed)
        while env.agents:
            env.step({agent: int(rng.integers(0, 5)) for agent in env.agents})
            steps += 1
    return steps / (time.perf_counter() - start)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episodes", type=int, default=1280)
    args = parser.parse_args()
    batched = batched_steps_per_second(args.episodes, batch=128)
    reference = reference_steps_per_second(args.episodes // 10)
    out = {
        "env": "simple_spread_v3",
        "batched_steps_per_s": batched,
        "mpe2_steps_per_s": reference,
        "speedup": batched / reference,
        "provenance": provenance(),
    }
    print(f"batched {batched:,.0f} steps/s   mpe2 {reference:,.0f} steps/s   speedup {out['speedup']:.0f}x")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "env_speed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
