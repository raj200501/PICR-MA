"""Cross-play: pair independently trained runs and measure team return.

For an ordered pair (ego A, partner B), slot 0 of the team is controlled by A's
policy and every other team slot by B's policy. In ``simple_adversary_v3`` the
adversary is A's co-trained adversary, so relative to A's self-play only the
teammates change. A == B gives self-play on the same evaluation seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from picr_ma.envs.mpe import make_env
from picr_ma.evaluation import eval_seeds, play_episodes
from picr_ma.networks import load_policy


@dataclass(frozen=True)
class RunRef:
    path: Path
    env: str
    method: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.method}/seed{self.seed}"


def find_runs(env_dir: str | Path) -> list[RunRef]:
    runs = []
    for summary_path in sorted(Path(env_dir).glob("*/seed*/summary.json")):
        summary = json.loads(summary_path.read_text())
        variant = summary_path.parent.parent.name
        runs.append(RunRef(summary_path.parent, summary["env"], variant, int(summary["seed"])))
    return runs


def _load(run: RunRef):
    adversary = run.path / "adversary.pt"
    return load_policy(run.path / "policy.pt"), (load_policy(adversary) if adversary.exists() else None)


def evaluate_ego(ego: RunRef, partners: list[RunRef], episodes: int) -> list[dict]:
    """Cross-play rows for one ego against every partner (including itself)."""
    spec = make_env(ego.env).spec
    ego_policy, ego_adversary = _load(ego)
    seeds = eval_seeds(episodes)
    rows = []
    for partner in partners:
        partner_policy = ego_policy if partner == ego else _load(partner)[0]
        team = [ego_policy] + [partner_policy] * (spec.n_team - 1)
        returns = play_episodes(ego.env, team, ego_adversary, seeds, sample_seed=0)["returns"]
        rows.append(
            {
                "env": ego.env,
                "ego_method": ego.method,
                "ego_seed": ego.seed,
                "partner_method": partner.method,
                "partner_seed": partner.seed,
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "episodes": episodes,
            }
        )
    return rows


def pair_matrix(rows: list[dict]) -> dict:
    """Index cross-play rows as {(ego_method, ego_seed, partner_method, partner_seed): return}."""
    return {
        (r["ego_method"], int(r["ego_seed"]), r["partner_method"], int(r["partner_seed"])): float(r["return_mean"]) for r in rows
    }


def seeds_of(rows: list[dict], method: str) -> list[int]:
    return sorted({int(r["ego_seed"]) for r in rows if r["ego_method"] == method})


def mean_cross_play(matrix: dict, ego_method: str, ego_seeds, partner_method: str, partner_seeds) -> float:
    """Mean return over ordered pairs of distinct runs (self-pairs excluded)."""
    values = [
        matrix[(ego_method, a, partner_method, b)]
        for a in ego_seeds
        for b in partner_seeds
        if not (ego_method == partner_method and a == b)
    ]
    return float(np.mean(values)) if values else float("nan")
