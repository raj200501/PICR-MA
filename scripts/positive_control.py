#!/usr/bin/env python
"""Instrument validation: would the evaluation detect conventions and order dependence if they existed?

    python scripts/positive_control.py

Uses scripted simple_spread_v3 policies (picr_ma/scripted.py) with known
properties and runs them through the same evaluation code as the learned
policies, on the same test episodes:

1. Convention incompatibility. A population of greedy-assignment teams, half
   claiming landmarks west-to-east and half east-to-west (with different
   amounts of action noise), is cross-played with the ego slot rotated. Mixed
   teams collide, so the population should show a clear self-play/cross-play
   gap and pair-specific residuals that replicate across halves of the
   episodes. A single-convention population is the comparison.
2. Order dependence. A policy that heads for the landmark nearest its
   first-listed teammate is evaluated under every per-agent teammate-order
   pattern. Relabeling patterns must leave its return unchanged (up to
   noise); genuine patterns must change it. The order-invariant greedy policy
   is unaffected by every pattern, exactly.

Writes results/positive_control.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from picr_ma.analysis.additive import additive_decomposition
from picr_ma.crossplay import order_shift_returns, returns_by_slot
from picr_ma.evaluation import eval_seeds
from picr_ma.scripted import FirstTeammateLandmark, GreedyAssignment

ROOT = Path(__file__).resolve().parents[1]
ENV = "simple_spread_v3"
NOISE = (0.02, 0.05, 0.1, 0.15)


def population_cross_play(population: dict[str, object], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for ego_name, ego in population.items():
        for partner_name, partner in population.items():
            returns = np.mean(returns_by_slot(ENV, ego, partner, None, seeds), axis=0)
            method, seed = ego_name.split("/")
            p_method, p_seed = partner_name.split("/")
            rows.append(
                {
                    "ego_method": method,
                    "ego_seed": int(seed),
                    "partner_method": p_method,
                    "partner_seed": int(p_seed),
                    "return_mean": float(returns.mean()),
                    "return_mean_half1": float(returns[0::2].mean()),
                    "return_mean_half2": float(returns[1::2].mean()),
                }
            )
    return pd.DataFrame(rows)


def gap(xp: pd.DataFrame) -> dict[str, float]:
    self_play = xp.ego_method.eq(xp.partner_method) & xp.ego_seed.eq(xp.partner_seed)
    return {"self_play": float(xp[self_play].return_mean.mean()), "cross_play": float(xp[~self_play].return_mean.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episodes", type=int, default=240)
    args = parser.parse_args()
    seeds = eval_seeds(args.episodes)
    out = {"env": ENV, "episodes": args.episodes}

    west = {f"west/{i}": GreedyAssignment("west", eps) for i, eps in enumerate(NOISE)}
    east = {f"east/{i}": GreedyAssignment("east", eps) for i, eps in enumerate(NOISE)}
    for name, population in [("one_convention", west), ("two_conventions", {**west, **east})]:
        xp = population_cross_play(population, seeds)
        out[name] = {**gap(xp), **additive_decomposition(xp)}
        out[name]["gap"] = out[name]["self_play"] - out[name]["cross_play"]
        print(
            f"{name:>16}: SP {out[name]['self_play']:.2f}  XP {out[name]['cross_play']:.2f}  gap {out[name]['gap']:.2f}  "
            f"additive R2 {out[name]['r2_additive']:.3f}  residual split-half r {out[name]['residual_split_half_corr']:.2f}"
        )

    for name, policy in [
        ("first_teammate_landmark", FirstTeammateLandmark(0, 0.05)),
        ("greedy_west", GreedyAssignment("west", 0.05)),
    ]:
        rows = pd.DataFrame(order_shift_returns(ENV, policy, None, seeds))
        mean_by_kind = rows.groupby("kind").return_mean.mean()
        base = mean_by_kind["identity"]
        out[name] = {
            "self_play": float(base),
            **{f"change_{kind}": float(mean_by_kind[kind] - base) for kind in ("relabeling", "genuine", "resampled")},
            "patterns": rows.to_dict(orient="records"),
        }
        print(
            f"{name:>24}: SP {base:.2f}  relabeling {out[name]['change_relabeling']:+.2f}  genuine {out[name]['change_genuine']:+.2f}  "
            f"resampled {out[name]['change_resampled']:+.2f}"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "positive_control.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
