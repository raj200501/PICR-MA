#!/usr/bin/env python
"""The IPPO hyperparameter sweep behind docs/tuning.md (IPPO only, tuning seed 1000).

    python scripts/tuning_sweep.py --workers 4

Each configuration trains one IPPO run and writes its summary to
results/tuning/ippo_sweep.jsonl (one JSON object per line, keyed by a tag
that encodes the configuration). The chosen configuration was additionally run
for 6M steps on simple_spread_v3 to check how training continues. Runs whose
tag is already in the file are skipped.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 1000
# (env, steps, lr, epochs, minibatches, episodes per update, entropy coefficient)
GRID = [
    ("simple_spread_v3", 3_000_000, 3e-4, 5, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 5, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 2e-3, 5, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 3e-4, 10, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 10, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 2e-3, 10, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 10, 4, 64, 0.0),
    ("simple_spread_v3", 3_000_000, 1e-3, 10, 4, 32, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 10, 8, 128, 0.01),
    ("simple_spread_v3", 3_000_000, 7e-4, 15, 4, 64, 0.01),
    ("simple_spread_v3", 3_000_000, 2e-3, 10, 8, 128, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 10, 16, 256, 0.01),
    ("simple_spread_v3", 3_000_000, 1e-3, 15, 8, 128, 0.01),
    ("simple_spread_v3", 6_000_000, 1e-3, 10, 8, 128, 0.01),
    ("simple_adversary_v3", 3_000_000, 1e-3, 10, 8, 128, 0.01),
    ("simple_adversary_v3", 3_000_000, 1e-3, 5, 4, 64, 0.01),
]


def tag(env, steps, lr, epochs, minibatches, num_envs, ent) -> str:
    return f"{env}_ippo_lr{lr}_ep{epochs}_mb{minibatches}_ne{num_envs}_ent{ent}_s{SEED}_steps{steps}"


def run(entry: tuple) -> dict:
    warnings.filterwarnings("ignore")
    from picr_ma.ppo import PPOConfig
    from picr_ma.train import TrainConfig, train

    env, steps, lr, epochs, minibatches, num_envs, ent = entry
    config = TrainConfig(
        env=env,
        method="ippo",
        seed=SEED,
        total_steps=steps,
        num_envs=num_envs,
        eval_episodes=500,
        ppo=PPOConfig(lr=lr, epochs=epochs, minibatches=minibatches, ent_coef=ent),
    )
    summary = train(config, ROOT / "results" / "tuning" / "runs" / tag(*entry))
    return {"tag": tag(*entry), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = ROOT / "results" / "tuning" / "ippo_sweep.jsonl"
    done = {json.loads(line)["tag"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [entry for entry in GRID if tag(*entry) not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as handle:
        for future in as_completed([pool.submit(run, entry) for entry in todo]):
            handle.write(json.dumps(future.result()) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
