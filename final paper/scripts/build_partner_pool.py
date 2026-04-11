#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.partner_pool import load_run_summaries


if __name__ == "__main__":
    summaries = load_run_summaries(ROOT / "results")
    output_path = ROOT / "results/partner_pool.json"
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(output_path)
