#!/usr/bin/env python
"""Self-play under shifted teammate order, for every run in a suite.

    python scripts/order_shift.py --suite main --episodes 240 --workers 4

Each run's team plays itself under every combination of per-agent teammate
orderings (relabeling patterns as a null control, genuinely different
conventions as the test) and with teammate features resampled from other
episodes, on the same evaluation episodes and adversary pool as cross-play.
Writes results/<suite>/<env>/order_shift.csv and order_shift.provenance.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate(args):
    warnings.filterwarnings("ignore")
    import torch

    from picr_ma.crossplay import evaluate_order_shift

    torch.set_num_threads(1)
    return evaluate_order_shift(*args)


def main() -> None:
    from picr_ma.crossplay import adversary_pool, find_runs
    from picr_ma.provenance import provenance

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default=None,
        help="evaluation episodes to use (default: validation for the lambda suite, test otherwise)",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    split = args.split or ("validation" if args.suite == "lambda" else "test")

    for env_dir in sorted(p for p in (args.results_dir / args.suite).iterdir() if p.is_dir()):
        runs = find_runs(env_dir)
        if not runs:
            continue
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            chunks = list(pool.map(_evaluate, [(run, args.episodes, adversary_pool(runs), split) for run in runs]))
        rows = [row for chunk in chunks for row in chunk]
        with (env_dir / "order_shift.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        meta = {"suite": args.suite, "episodes": args.episodes, "split": split, "provenance": provenance()}
        (env_dir / "order_shift.provenance.json").write_text(json.dumps(meta, indent=2))
        print(f"{env_dir.name}: {len(runs)} runs -> {env_dir / 'order_shift.csv'}", flush=True)


if __name__ == "__main__":
    main()
