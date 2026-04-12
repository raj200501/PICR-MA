#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    results_dir = ROOT / "results"
    profile_path = results_dir / "ACTIVE_PROFILE.txt"
    profile = profile_path.read_text().strip() if profile_path.exists() else "UNKNOWN"
    summaries = sorted(results_dir.glob("run_*/summary.json"))
    payload = {
        "profile": profile,
        "runs": len(summaries),
        "run_ids": [path.parent.name for path in summaries],
    }
    print(json.dumps(payload, indent=2))
