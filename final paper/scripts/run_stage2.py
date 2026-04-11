#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import run_experiment


if __name__ == "__main__":
    profile = (ROOT / "results/ACTIVE_PROFILE.txt").read_text().strip() if (ROOT / "results/ACTIVE_PROFILE.txt").exists() else "UNKNOWN"
    if profile not in {"PROFILE_FULL_SAFE", "PROFILE_LEAN_SAFE"}:
        print(f"Stage2 skipped under profile {profile}.")
    else:
        for value in [0.05, 0.2, 0.5]:
            run_experiment(
                env="simple_spread_v3",
                method="picr_ma",
                seed=0,
                steps=800,
                tag=f"stage2_lambda_{value}",
                max_cycles=25,
                picr_lambda=value,
            )
        print("stage2 done")
