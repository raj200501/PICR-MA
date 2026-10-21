#!/usr/bin/env python
"""Evaluate all ordered pairs of runs in a suite, per environment.

    python scripts/crossplay.py --suite main --episodes 32 --workers 4

Writes results/<suite>/<env>/crossplay.csv with one row per (ego, partner).
"""

from __future__ import annotations

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate(args):
    warnings.filterwarnings("ignore")
    import torch

    from picr_ma.crossplay import evaluate_ego

    torch.set_num_threads(1)
    return evaluate_ego(*args)


def main() -> None:
    from picr_ma.crossplay import find_runs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for env_dir in sorted(p for p in (args.results_dir / args.suite).iterdir() if p.is_dir()):
        runs = find_runs(env_dir)
        if not runs:
            continue
        print(f"{env_dir.name}: {len(runs)} runs, {len(runs) ** 2} ordered pairs", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            chunks = list(pool.map(_evaluate, [(ego, runs, args.episodes) for ego in runs]))
        rows = [row for chunk in chunks for row in chunk]
        out = env_dir / "crossplay.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}", flush=True)


if __name__ == "__main__":
    main()
