#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_run
from scripts.train import run_experiment


STAGE1_ENVS = [
    ("matrix_coordination", 600, None),
    ("matrix_staghunt", 600, None),
    ("simple_spread_v3", 1000, 25),
    ("simple_adversary_v3", 1000, 25),
]


if __name__ == "__main__":
    created = []
    for env_name, steps, max_cycles in STAGE1_ENVS:
        for method in ["ippo", "picr_ma", "deepsets", "otherplay"]:
            for seed in [0, 1]:
                run_id = run_experiment(
                    env=env_name,
                    method=method,
                    seed=seed,
                    steps=steps,
                    tag="stage1",
                    max_cycles=max_cycles,
                )
                created.append(ROOT / "results" / run_id)
    for run_dir in created:
        evaluate_run(run_dir, ROOT / "results")
    print("stage1 done")
