#!/usr/bin/env python
"""Print the IPPO hyperparameter sweep (results/tuning/ippo_sweep.jsonl) as a Markdown table."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r"(\w+?_v3)_ippo_lr([\d.e-]+)_ep(\d+)_mb(\d+)_ne(\d+)_ent([\d.]+)_s(\d+)")


def main() -> None:
    rows = []
    for line in (ROOT / "results" / "tuning" / "ippo_sweep.jsonl").read_text().splitlines():
        run = json.loads(line)
        env, lr, epochs, minibatches, envs, ent, _ = TAG.match(run["tag"]).groups()
        steps = round(run["env_steps"] / 1e6)
        rows.append((env, steps, -run["self_play_return"], float(lr), int(epochs), int(minibatches), int(envs), float(ent), run))
    rows.sort(key=lambda r: (r[0] != "simple_spread_v3", r[1], r[2]))
    print("| Env | Steps | lr | Epochs | Minibatches | Envs / update | Entropy coef. | Self-play return | Perm. JS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for env, steps, _, lr, epochs, minibatches, envs, ent, run in rows:
        print(
            f"| {env} | {steps}M | {lr:g} | {epochs} | {minibatches} | {envs} | {ent:g} | "
            f"{run['self_play_return']:.2f} | {run['perm_js']:.3f} |"
        )


if __name__ == "__main__":
    main()
