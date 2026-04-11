#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.matrix_games import make_matrix_env
from src.envs.mpe_wrappers import make_mpe_env, simple_adversary_goal_margin, simple_spread_collisions
from src.eval.crossplay import load_policy_bundle
from src.eval.partner_pool import load_run_summaries, partner_pool
from src.utils.io_utils import read_json, write_json
from src.utils.serialization import to_csv_gz


def build_env(env_name: str, seed: int):
    if env_name.startswith("matrix_"):
        env = make_matrix_env(env_name, seed=seed)
        spec = type(
            "Spec",
            (),
            {
                "cooperative_agents": tuple(env.agents),
                "adversary_agents": (),
                "teammate_slicing": env.config.teammate_slicing,
            },
        )()
        return env, spec
    return make_mpe_env(env_name, seed=seed, max_cycles=25)


def evaluate_pair(ego_bundle: dict, partner_bundle: dict, episodes: int = 8) -> dict[str, float]:
    env_name = ego_bundle["summary"]["env"]
    rng = np.random.default_rng(ego_bundle["summary"]["seed"] * 1000 + partner_bundle["summary"]["seed"])
    returns = []
    successes = []
    collisions = []
    goal_margins = []
    for episode in range(episodes):
        env, spec = build_env(env_name, seed=int(rng.integers(0, 1_000_000)))
        observations, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        episode_return = 0.0
        while not done:
            action_dict = {}
            for index, agent in enumerate(spec.cooperative_agents):
                obs = np.asarray([observations[agent]], dtype=np.float32)
                policy = ego_bundle["cooperative_policy"] if index == 0 else partner_bundle["cooperative_policy"]
                action, _, _ = policy.act(obs, rng=rng, greedy=True)
                action_dict[agent] = int(action[0])
            for agent in spec.adversary_agents:
                obs = np.asarray([observations[agent]], dtype=np.float32)
                adversary_policy = ego_bundle.get("adversary_policy")
                if adversary_policy is None:
                    action_dict[agent] = 0
                else:
                    action, _, _ = adversary_policy.act(obs, rng=rng, greedy=True)
                    action_dict[agent] = int(action[0])
            observations, rewards, terminateds, truncations, infos = env.step(action_dict)
            done = all(terminateds.values()) or all(truncations.values())
            episode_return += float(np.mean([rewards[a] for a in spec.cooperative_agents]))
            if env_name == "simple_spread_v3":
                collisions.append(simple_spread_collisions(env))
            if env_name == "simple_adversary_v3":
                goal_margins.append(simple_adversary_goal_margin(env))
            if env_name.startswith("matrix_"):
                successes.append(float(np.mean([infos[a]["success"] for a in spec.cooperative_agents])))
        returns.append(episode_return)
    return {
        "crossplay_return": float(np.mean(returns)) if returns else 0.0,
        "success_mean": float(np.mean(successes)) if successes else 0.0,
        "collision_mean": float(np.mean(collisions)) if collisions else 0.0,
        "goal_margin_mean": float(np.mean(goal_margins)) if goal_margins else 0.0,
    }


def evaluate_run(
    run_dir: str | Path,
    results_dir: str | Path = "results",
    partner_run_tag: str | None = None,
    episodes: int = 8,
) -> list[dict]:
    run_path = Path(run_dir)
    summary = read_json(run_path / "summary.json")
    env_name = summary["env"]
    ego_bundle = load_policy_bundle(run_path, slicing=build_env(env_name, seed=summary["seed"])[1].teammate_slicing)
    pool_tag = partner_run_tag or summary.get("run_tag")
    partner_rows = partner_pool(
        results_dir=results_dir,
        env=env_name,
        exclude_run_id=summary["run_id"],
        run_tag=pool_tag,
    )
    rows = []
    for partner_summary in partner_rows:
        partner_bundle = load_policy_bundle(
            partner_summary["run_dir"],
            slicing=build_env(env_name, seed=partner_summary["seed"])[1].teammate_slicing,
        )
        metrics = evaluate_pair(ego_bundle, partner_bundle, episodes=episodes)
        rows.append(
            {
                "ego_run_id": summary["run_id"],
                "ego_method": summary["method"],
                "ego_seed": summary["seed"],
                "partner_run_id": partner_summary["run_id"],
                "partner_method": partner_summary["method"],
                "partner_seed": partner_summary["seed"],
                "crossplay_return": metrics["crossplay_return"],
                "success_mean": metrics["success_mean"],
                "collision_mean": metrics["collision_mean"],
                "goal_margin_mean": metrics["goal_margin_mean"],
            }
        )

    to_csv_gz(rows, run_path / "eval_crossplay.csv.gz")
    same_method_rows = [row for row in rows if row["partner_method"] == summary["method"]]
    summary["crossplay_pool_run_tag"] = pool_tag
    summary["crossplay_partner_count"] = len(rows)
    summary["crossplay_return"] = (
        float(np.mean([row["crossplay_return"] for row in same_method_rows]))
        if same_method_rows
        else None
    )
    write_json(run_path / "summary.json", summary)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--partner-run-tag", default=None)
    parser.add_argument("--episodes", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_dir:
        evaluate_run(
            args.run_dir,
            args.results_dir,
            partner_run_tag=args.partner_run_tag,
            episodes=args.episodes,
        )
    else:
        for summary in load_run_summaries(args.results_dir):
            evaluate_run(
                summary["run_dir"],
                args.results_dir,
                partner_run_tag=args.partner_run_tag,
                episodes=args.episodes,
            )
