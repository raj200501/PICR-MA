#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    rows = []
    for run_dir in sorted((ROOT / "results").glob("run_*")):
        size_mb = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()) / 1024**2
        rows.append((run_dir.name, round(size_mb, 4)))
    with (ROOT / "results/artifact_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "artifact_mb"])
        writer.writerows(rows)
    print("inventory written")
