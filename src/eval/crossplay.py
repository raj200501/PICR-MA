from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.actor_critic import load_actor_critic
from src.utils.io_utils import read_json
from src.utils.serialization import load_npz


def load_policy_bundle(run_dir: str | Path, slicing=None) -> dict:
    run_path = Path(run_dir)
    summary = read_json(run_path / "summary.json")
    cooperative_state = load_npz(run_path / "best.ckpt")
    bundle = {
        "summary": summary,
        "cooperative_policy": load_actor_critic(cooperative_state, slicing=slicing),
    }
    adversary_path = run_path / "best_adversary.ckpt"
    if adversary_path.exists():
        bundle["adversary_policy"] = load_actor_critic(load_npz(adversary_path))
    return bundle


def crossplay_pairings(
    ego_run: dict,
    partner_runs: list[dict],
    max_partners: int | None = None,
) -> list[dict]:
    rows = []
    for partner in partner_runs[:max_partners]:
        rows.append(
            {
                "ego_run_id": ego_run["run_id"],
                "partner_run_id": partner["run_id"],
                "ego_method": ego_run["method"],
                "partner_method": partner["method"],
                "env": ego_run["env"],
            }
        )
    return rows


def crossplay_score(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return {"mean": 0.0, "se": 0.0}
    return {
        "mean": float(arr.mean()),
        "se": float(arr.std(ddof=0) / max(len(arr) ** 0.5, 1.0)),
    }
