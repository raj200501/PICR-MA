"""Cross-play: pair independently trained runs and measure team return.

For an ordered pair (ego A, partner B), one team slot is controlled by A's
policy and the other slots by B's. Because MPE lists teammates in index order,
the slot matters: it determines where A appears in B's observations (first or
second teammate). Every pairing is therefore scored with A in each slot in turn,
on the same evaluation episodes, and averaged. A == B is self-play.

In ``simple_adversary_v3`` each run co-trains its own adversary, so scoring a
team against its own adversary would compare methods against different
opponents. Instead every pairing (self-play included) is scored against the
same pool: the adversaries of all runs in the suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from picr_ma.envs.mpe import make_env
from picr_ma.envs.slicing import view_patterns
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


def _load_pool(adversaries: list[RunRef], env: str):
    pool = [load_policy(run.path / "adversary.pt") for run in adversaries] or None
    if make_env(env).spec.n_adversaries and pool is None:
        raise ValueError(f"{env} evaluation needs a non-empty adversary pool")
    return pool


def _split_halves(returns: np.ndarray) -> tuple[float, float]:
    """Means over even and odd episodes. Adversaries are assigned to episodes in
    pairs, so both halves face the same opponents."""
    return float(returns[0::2].mean()), float(returns[1::2].mean())


def returns_by_slot(env: str, ego_policy, partner_policy, pool, seeds: list[int]) -> list[np.ndarray]:
    """Per-episode team returns with the ego policy in each team slot in turn (partner in the rest)."""
    n_team = make_env(env).spec.n_team
    if ego_policy is partner_policy:
        return [play_episodes(env, [ego_policy] * n_team, pool, seeds, sample_seed=0)["returns"]] * n_team
    per_slot = []
    for slot in range(n_team):
        team = [partner_policy] * n_team
        team[slot] = ego_policy
        per_slot.append(play_episodes(env, team, pool, seeds, sample_seed=0)["returns"])
    return per_slot


def evaluate_ego(
    ego: RunRef,
    partners: list[RunRef],
    episodes: int,
    adversaries: list[RunRef] = (),
    split: str = "test",
) -> list[dict]:
    """Cross-play rows for one ego against every partner (including itself), ego slot rotated."""
    ego_policy = load_policy(ego.path / "policy.pt")
    pool = _load_pool(adversaries, ego.env)
    seeds = eval_seeds(episodes, split)
    rows = []
    for partner in partners:
        partner_policy = ego_policy if partner == ego else load_policy(partner.path / "policy.pt")
        per_slot = returns_by_slot(ego.env, ego_policy, partner_policy, pool, seeds)
        returns = np.mean(per_slot, axis=0)  # per episode, averaged over the ego's slot
        half1, half2 = _split_halves(returns)
        rows.append(
            {
                "env": ego.env,
                "ego_method": ego.method,
                "ego_seed": ego.seed,
                "partner_method": partner.method,
                "partner_seed": partner.seed,
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "return_mean_half1": half1,
                "return_mean_half2": half2,
                **{f"return_mean_slot{k}": float(r.mean()) for k, r in enumerate(per_slot)},
                "episodes": episodes,
                "adversary_pool_size": len(adversaries),
                "split": split,
            }
        )
    return rows


def evaluate_order_shift(run: RunRef, episodes: int, adversaries: list[RunRef] = (), split: str = "test") -> list[dict]:
    """Self-play of one run under every per-agent teammate ordering, plus a resampling control.

    Each agent's view of its teammates is reordered by its own permutation. A
    pattern that a renaming of the agents also produces (``relabeling``) cannot
    change the expected return of a shared policy with exchangeable agents, so
    those rows are a null control; ``genuine`` patterns (the cyclic
    conventions) test whether the policy's returns depend on teammate order.
    The ``resampled`` row replaces teammate features with those of another
    episode, showing how much the policy relies on teammate information at all.
    Same episodes, sampling seed and adversary pool as cross-play, so the
    identity row equals the cross-play diagonal.
    """
    policy = load_policy(run.path / "policy.pt")
    pool = _load_pool(adversaries, run.env)
    rows = order_shift_returns(run.env, policy, pool, eval_seeds(episodes, split))
    return [{"env": run.env, "method": run.method, "seed": run.seed, **row, "episodes": episodes, "split": split} for row in rows]


def order_shift_returns(env: str, policy, pool, seeds: list[int]) -> list[dict]:
    """Mean self-play return under every per-agent teammate-order pattern and with teammates resampled."""
    spec = make_env(env).spec
    conditions = [(p["kind"], p["slot_perms"], False) for p in view_patterns(spec.n_team, spec.slicing)]
    conditions.append(("resampled", None, True))
    rows = []
    for kind, slot_perms, resample in conditions:
        returns = play_episodes(
            env, [policy] * spec.n_team, pool, seeds, sample_seed=0, slot_permutations=slot_perms, resample_teammates=resample
        )["returns"]
        pattern = "none" if slot_perms is None else " ".join("".join(map(str, p)) for p in slot_perms)
        rows.append({"kind": kind, "pattern": pattern, "return_mean": float(returns.mean())})
    return rows
