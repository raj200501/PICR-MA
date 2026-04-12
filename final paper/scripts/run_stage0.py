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
    run_experiment(env="matrix_coordination", method="ippo", seed=0, steps=120, tag="stage0")
    run_experiment(env="matrix_coordination", method="picr_ma", seed=0, steps=120, tag="stage0")
    run_experiment(env="simple_spread_v3", method="ippo", seed=0, steps=200, tag="stage0", max_cycles=10)
    evaluate_run(ROOT / "results/run_stage0_matrix_coordination_ippo_s0", ROOT / "results")
    print("stage0 done")
