#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_run
from scripts.train import run_experiment


if __name__ == "__main__":
    run_experiment(env="matrix_coordination", method="ippo", seed=0, steps=60, tag="smoke")
    run_experiment(env="matrix_coordination", method="ippo", seed=1, steps=60, tag="smoke")
    run_experiment(env="simple_spread_v3", method="ippo", seed=0, steps=100, tag="smoke", max_cycles=10)
    evaluate_run(ROOT / "results/run_smoke_matrix_coordination_ippo_s0", ROOT / "results")
    print("smoke ok")
