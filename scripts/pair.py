#!/usr/bin/env python
"""Score two committed runs as a team: each run's self-play and their cross-play.

    python scripts/pair.py results/main/simple_spread_v3/ippo/seed0 results/main/simple_spread_v3/picr/seed3

This is the evaluation behind crossplay.csv, for a single pair: the same test
episodes and random numbers, the ego seated in each slot in turn, and in
simple_adversary_v3 the suite's shared pool of adversaries. The numbers it
prints therefore match the corresponding rows of crossplay.csv. About a minute
on one CPU core.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


def _run_ref(path: Path):
    from picr_ma.crossplay import RunRef

    summary = json.loads((path / "summary.json").read_text())
    return RunRef(path, summary["env"], path.parent.name, int(summary["seed"]))


def main() -> None:
    warnings.filterwarnings("ignore")
    from picr_ma.crossplay import adversary_pool, evaluate_ego, find_runs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path, help="a run directory, e.g. results/main/simple_spread_v3/ippo/seed0")
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args()
    a, b = _run_ref(args.run_a.resolve()), _run_ref(args.run_b.resolve())
    if a.env != b.env:
        raise SystemExit(f"the runs are from different environments ({a.env}, {b.env})")
    pool = adversary_pool(find_runs(a.path.parent.parent))
    rows = evaluate_ego(a, [a, b], args.episodes, pool, args.split) + evaluate_ego(b, [b], args.episodes, pool, args.split)
    rows += evaluate_ego(b, [a], args.episodes, pool, args.split)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{a.env}: {args.episodes} {args.split} episodes" + (f", {len(pool)} pooled adversaries" if pool else ""))
    for row in rows:
        ego, partner = f"{row['ego_method']}/seed{row['ego_seed']}", f"{row['partner_method']}/seed{row['partner_seed']}"
        label = f"{ego} self-play" if ego == partner else f"{ego} with two {partner}"
        slots = "  ".join(f"{row[f'return_mean_slot{k}']:7.2f}" for k in range(3))
        print(f"  {label:<48} {row['return_mean']:7.2f}   by ego slot: {slots}")


if __name__ == "__main__":
    main()
