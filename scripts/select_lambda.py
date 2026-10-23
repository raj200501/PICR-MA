#!/usr/bin/env python
"""Select the PICR weight for the main suite from the lambda suite.

    python scripts/select_lambda.py

Rule (fixed before any lambda-suite run was trained): for each environment,
choose the lambda with the highest mean same-variant cross-play return on the
selection seeds. Ties are broken toward the smaller lambda. The main suite's
evaluation seeds are never used. Writes results/lambda/selection.json; the
chosen values are copied into ``picr_lambda_by_env`` in configs/experiments.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from picr_ma.analysis.aggregate import crossplay_block, load_crossplay, load_summaries

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    suite_dir = parser.parse_args().results_dir / "lambda"
    summaries = load_summaries(suite_dir)
    selection = {}
    for env in sorted(set(summaries.env)):
        xp = load_crossplay(suite_dir, env)
        scores = {}
        for variant in sorted(set(summaries[summaries.env == env].variant)):
            block = crossplay_block(xp, variant, variant)
            scores[float(variant.split("_lam")[1])] = float(block[~np.eye(len(block), dtype=bool)].mean())
        best = max(sorted(scores), key=lambda lam: scores[lam])
        selection[env] = {"selected_lambda": best, "cross_play_by_lambda": scores}
        print(f"{env}: lambda={best:g}   " + "  ".join(f"{lam:g}: {v:.2f}" for lam, v in sorted(scores.items())))
    (suite_dir / "selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
