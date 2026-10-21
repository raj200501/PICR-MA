#!/usr/bin/env python
"""Train every run in a suite from configs/experiments.yaml, in parallel.

    python scripts/run_experiments.py --suite main --workers 4
    python scripts/run_experiments.py --suite main --dry-run

Finished runs (those with summary.json) are skipped, so the command is safe to
re-run after an interruption.
"""

from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _execute(run):
    warnings.filterwarnings("ignore")
    from picr_ma.experiments import execute

    return execute(run)


def main() -> None:
    from picr_ma.experiments import expand_suite, load_suites

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--config", default=ROOT / "configs" / "experiments.yaml")
    parser.add_argument("--results-dir", default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = expand_suite(load_suites(args.config), args.suite, args.results_dir)
    todo = [r for r in runs if not r.done]
    print(f"suite {args.suite}: {len(runs)} runs, {len(runs) - len(todo)} done, {len(todo)} to train", flush=True)
    if args.dry_run:
        for run in todo:
            print(f"  {run.out_dir.relative_to(args.results_dir)}")
        return
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_execute, run): run for run in todo}
        for i, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            summary = future.result()
            print(
                f"[{i}/{len(todo)} {time.time() - start:7.0f}s] {run.out_dir.relative_to(args.results_dir)} "
                f"return={summary['self_play_return']:.2f} perm_js={summary['perm_js']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
