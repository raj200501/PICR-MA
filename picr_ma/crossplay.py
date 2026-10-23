"""Cross-play: pair independently trained runs and measure team return.

For an ordered pair (ego A, partner B), team slot 0 is controlled by A's policy
and every other team slot by B's policy; A == B is self-play. Every pairing is
scored on the same reserved evaluation seeds.

In ``simple_adversary_v3`` each run co-trains its own adversary, so scoring a
team against its own adversary would compare methods against different
opponents. Instead every pairing (self-play included) is scored against the
same pool: the adversaries of all runs in the suite, assigned to evaluation
episodes round-robin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def adversary_pool(runs: list[RunRef]) -> list[RunRef]:
    """Runs whose co-trained adversary joins the shared evaluation pool."""
    return [run for run in runs if (run.path / "adversary.pt").exists()]


def evaluate_ego(ego: RunRef, partners: list[RunRef], episodes: int, adversaries: list[RunRef] = ()) -> list[dict]:
    """Cross-play rows for one ego against every partner (including itself)."""
    spec = make_env(ego.env).spec
    ego_policy = load_policy(ego.path / "policy.pt")
    adversary = [load_policy(run.path / "adversary.pt") for run in adversaries] or None
    if spec.n_adversaries and adversary is None:
        raise ValueError(f"{ego.env} cross-play needs a non-empty adversary pool")
    seeds = eval_seeds(episodes)
    rows = []
    for partner in partners:
        partner_policy = ego_policy if partner == ego else load_policy(partner.path / "policy.pt")
        team = [ego_policy] + [partner_policy] * (spec.n_team - 1)
        returns = play_episodes(ego.env, team, adversary, seeds, sample_seed=0)["returns"]
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
                "adversary_pool_size": len(adversaries),
            }
        )
    return rows
