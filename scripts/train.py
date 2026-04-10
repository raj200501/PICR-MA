#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos.ppo import PPOConfig, compute_returns_and_advantages, ppo_update
from src.algos.rollout_buffer import RolloutBatch
from src.envs.matrix_games import make_matrix_env
from src.envs.mpe_wrappers import (
    make_mpe_env,
    simple_adversary_goal_margin,
    simple_spread_collisions,
)
from src.eval.permutation_sensitivity import sensitivity
from src.methods.otherplay import permute_observation_batch, teammate_permutation
from src.models.actor_critic import build_actor_critic
from src.utils.config import dump_yaml
from src.utils.io_utils import ensure_dir, write_json, write_text
from src.utils.logging_utils import log
from src.utils.serialization import save_npz, to_csv_gz


@dataclass
class ExperimentConfig:
    env: str
    method: str
    seed: int
    profile: str
    total_env_steps: int
    max_cycles: int
    hidden_dim: int = 64
    learning_rate: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    picr_lambda: float = 0.2
    teammate_dropout_p: float = 0.0
    eval_episodes: int = 8
    run_tag: str = "manual"


def active_profile() -> str:
    profile_path = Path("results/ACTIVE_PROFILE.txt")
    return profile_path.read_text().strip() if profile_path.exists() else "PROFILE_SCAFFOLD_ONLY"


def safe_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ["numpy", "pandas", "matplotlib", "pettingzoo", "gymnasium", "supersuit", "pygame"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def parameter_count(model) -> int:
    return int(sum(value.size for value in model.parameters().values()))


def make_run_id(config: ExperimentConfig) -> str:
    return f"run_{config.run_tag}_{config.env}_{config.method}_s{config.seed}"


def build_experiment_env(config: ExperimentConfig, seed: int):
    if config.env.startswith("matrix_"):
        env = make_matrix_env(config.env, seed=seed)
        spec = type(
            "Spec",
            (),
            {
                "env_name": config.env,
                "action_dim": env.config.action_dim,
                "obs_dim": env.config.obs_dim,
                "cooperative_agents": tuple(env.agents),
                "adversary_agents": (),
                "teammate_slicing": env.config.teammate_slicing,
                "max_cycles": env.config.max_steps,
            },
        )()
        return env, spec
    return make_mpe_env(config.env, seed=seed, max_cycles=config.max_cycles)


def choose_policy_kind(method: str) -> str:
    return "deepsets" if method == "deepsets" else "mlp"


def preprocess_observations(
    method: str,
    observations: np.ndarray,
    slicing,
    rng: np.random.Generator,
    permutation: tuple[int, ...] | None = None,
) -> np.ndarray:
    if method == "otherplay" and slicing.n_teammates > 1:
        return permute_observation_batch(observations, slicing, rng, permutation=permutation)
    return np.asarray(observations, dtype=np.float32)


def evaluate_permutation_metric(policy, obs_samples: list[np.ndarray], slicing) -> tuple[list[dict], float]:
    if not obs_samples:
        return [], 0.0
    rows = []
    values = []
    for index, obs in enumerate(obs_samples):
        if slicing.n_teammates <= 1:
            score = 0.0
            per_perm = [0.0]
        else:
            score, per_perm = sensitivity(policy.action_distribution, obs, slicing)
        values.append(score)
        rows.append(
            {
                "sample_index": index,
                "permutation_js": float(score),
                "num_permutations": len(per_perm),
            }
        )
    return rows, float(np.mean(values))


def update_from_buffer(
    policy,
    buffer: RolloutBatch,
    rng: np.random.Generator,
    ppo_config: PPOConfig,
    slicing=None,
) -> dict[str, float]:
    arrays = buffer.as_arrays()
    arrays["returns"], arrays["advantages"] = compute_returns_and_advantages(
        rewards=arrays["rewards"],
        dones=arrays["dones"],
        values=arrays["values"],
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )
    stats = ppo_update(
        policy=policy,
        batch=arrays,
        rng=rng,
        config=ppo_config,
        slicing=slicing,
    )
    buffer.clear()
    return stats


def evaluate_in_team(
    config: ExperimentConfig,
    cooperative_policy,
    adversary_policy=None,
    episodes: int = 8,
) -> dict[str, float]:
    rng = np.random.default_rng(config.seed + 999)
    returns = []
    collisions = []
    goal_margins = []
    successes = []
    for episode in range(episodes):
        env, spec = build_experiment_env(config, seed=config.seed + 5000 + episode)
        observations, _ = env.reset(seed=config.seed + 5000 + episode)
        episode_return = 0.0
        episode_success = []
        done = False
        while not done:
            coop_obs = np.stack([observations[agent] for agent in spec.cooperative_agents], axis=0)
            coop_actions, _, _ = cooperative_policy.act(coop_obs, rng=rng, greedy=True)
            action_dict = {
                agent: int(action)
                for agent, action in zip(spec.cooperative_agents, coop_actions)
            }
            if spec.adversary_agents:
                adv_obs = np.stack([observations[agent] for agent in spec.adversary_agents], axis=0)
                if adversary_policy is None:
                    adv_actions = np.zeros(len(spec.adversary_agents), dtype=np.int64)
                else:
                    adv_actions, _, _ = adversary_policy.act(adv_obs, rng=rng, greedy=True)
                for agent, action in zip(spec.adversary_agents, adv_actions):
                    action_dict[agent] = int(action)
            observations, rewards, terminateds, truncations, infos = env.step(action_dict)
            done = all(terminateds.values()) or all(truncations.values())
            coop_reward = float(np.mean([rewards[agent] for agent in spec.cooperative_agents]))
            episode_return += coop_reward
            if config.env == "simple_spread_v3":
                collisions.append(simple_spread_collisions(env))
            if config.env == "simple_adversary_v3":
                goal_margins.append(simple_adversary_goal_margin(env))
            if config.env.startswith("matrix_"):
                episode_success.append(float(np.mean([infos[a]["success"] for a in spec.cooperative_agents])))
        returns.append(episode_return)
        if episode_success:
            successes.append(float(np.mean(episode_success)))
    result = {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "collision_mean": float(np.mean(collisions)) if collisions else 0.0,
        "goal_margin_mean": float(np.mean(goal_margins)) if goal_margins else 0.0,
        "success_mean": float(np.mean(successes)) if successes else 0.0,
    }
    return result


def run_experiment(
    env: str = "matrix_coordination",
    method: str = "ippo",
    seed: int = 0,
    steps: int = 500,
    tag: str = "manual",
    max_cycles: int | None = None,
    picr_lambda: float = 0.2,
    teammate_dropout_p: float = 0.0,
    eval_episodes: int = 8,
) -> str:
    profile = active_profile()
    config = ExperimentConfig(
        env=env,
        method=method,
        seed=seed,
        profile=profile,
        total_env_steps=steps,
        max_cycles=max_cycles or (1 if env.startswith("matrix_") else 25),
        picr_lambda=picr_lambda,
        teammate_dropout_p=teammate_dropout_p,
        eval_episodes=eval_episodes,
        run_tag=tag,
    )
    run_id = make_run_id(config)
    out_dir = ensure_dir(Path("results") / run_id)
    rng = np.random.default_rng(seed)
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        update_epochs=config.update_epochs,
        picr_lambda=config.picr_lambda if method == "picr_ma" else 0.0,
        teammate_dropout_p=config.teammate_dropout_p,
    )

    env_instance, spec = build_experiment_env(config, seed=config.seed)
    cooperative_policy = build_actor_critic(
        kind=choose_policy_kind(method),
        input_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        seed=seed,
        slicing=spec.teammate_slicing,
        hidden_dim=config.hidden_dim,
    )
    adversary_policy = None
    if spec.adversary_agents:
        adversary_obs_dim = int(env_instance.observation_space(spec.adversary_agents[0]).shape[0])
        adversary_policy = build_actor_critic(
            kind="mlp",
            input_dim=adversary_obs_dim,
            action_dim=spec.action_dim,
            seed=seed + 1000,
            hidden_dim=config.hidden_dim,
        )

    cooperative_buffer = RolloutBatch()
    adversary_buffer = RolloutBatch()
    train_rows = []
    permutation_obs_samples = []
    best_return = -float("inf")
    start_time = time.time()
    env_steps = 0
    episode_index = 0
    run_log_lines = []

    while env_steps < config.total_env_steps:
        observations, _ = env_instance.reset(seed=config.seed + episode_index)
        episode_return = 0.0
        episode_success = []
        episode_teammate_permutation = None
        if method == "otherplay" and spec.teammate_slicing.n_teammates > 1:
            episode_teammate_permutation = teammate_permutation(rng, spec.teammate_slicing.n_teammates)
        done = False
        while not done and env_steps < config.total_env_steps:
            coop_raw = np.stack([observations[agent] for agent in spec.cooperative_agents], axis=0)
            coop_policy_obs = preprocess_observations(
                method,
                coop_raw,
                spec.teammate_slicing,
                rng,
                permutation=episode_teammate_permutation,
            )
            coop_actions, coop_log_probs, coop_values = cooperative_policy.act(coop_policy_obs, rng)
            action_dict = {
                agent: int(action)
                for agent, action in zip(spec.cooperative_agents, coop_actions)
            }

            if adversary_policy is not None:
                adv_obs = np.stack([observations[agent] for agent in spec.adversary_agents], axis=0)
                adv_actions, adv_log_probs, adv_values = adversary_policy.act(adv_obs, rng)
                for agent, action in zip(spec.adversary_agents, adv_actions):
                    action_dict[agent] = int(action)
            else:
                adv_obs = np.zeros((0, 0), dtype=np.float32)
                adv_log_probs = np.zeros(0, dtype=np.float32)
                adv_values = np.zeros(0, dtype=np.float32)

            next_obs, rewards, terminateds, truncations, infos = env_instance.step(action_dict)
            done = all(terminateds.values()) or all(truncations.values())
            coop_rewards = np.array([rewards[agent] for agent in spec.cooperative_agents], dtype=np.float32)
            coop_done = np.array([terminateds[agent] or truncations[agent] for agent in spec.cooperative_agents])

            for index in range(len(spec.cooperative_agents)):
                cooperative_buffer.append(
                    observation=coop_policy_obs[index],
                    raw_observation=coop_raw[index],
                    action=int(coop_actions[index]),
                    reward=float(coop_rewards[index]),
                    done=bool(coop_done[index]),
                    log_prob=float(coop_log_probs[index]),
                    value=float(coop_values[index]),
                )
                if len(permutation_obs_samples) < 64:
                    permutation_obs_samples.append(coop_raw[index].copy())

            if adversary_policy is not None:
                adv_rewards = np.array([rewards[agent] for agent in spec.adversary_agents], dtype=np.float32)
                adv_done = np.array([terminateds[agent] or truncations[agent] for agent in spec.adversary_agents])
                for index in range(len(spec.adversary_agents)):
                    adversary_buffer.append(
                        observation=adv_obs[index],
                        raw_observation=adv_obs[index],
                        action=int(adv_actions[index]),
                        reward=float(adv_rewards[index]),
                        done=bool(adv_done[index]),
                        log_prob=float(adv_log_probs[index]),
                        value=float(adv_values[index]),
                    )

            episode_return += float(np.mean(coop_rewards))
            if config.env.startswith("matrix_"):
                episode_success.append(float(np.mean([infos[a]["success"] for a in spec.cooperative_agents])))
            observations = next_obs
            env_steps += 1

        coop_update = update_from_buffer(
            policy=cooperative_policy,
            buffer=cooperative_buffer,
            rng=rng,
            ppo_config=ppo_config,
            slicing=spec.teammate_slicing,
        )
        adv_update = {"policy_loss": 0.0, "value_loss": 0.0}
        if adversary_policy is not None and adversary_buffer.size() > 0:
            adv_update = update_from_buffer(
                policy=adversary_policy,
                buffer=adversary_buffer,
                rng=rng,
                ppo_config=PPOConfig(
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    clip_ratio=config.clip_ratio,
                    update_epochs=config.update_epochs,
                    picr_lambda=0.0,
                ),
            )

        metrics = {
            "episode": episode_index,
            "env_step": env_steps,
            "episodic_return": round(episode_return, 6),
            "success_rate": round(float(np.mean(episode_success)) if episode_success else 0.0, 6),
            "policy_loss": round(coop_update["policy_loss"], 6),
            "value_loss": round(coop_update["value_loss"], 6),
            "picr_loss": round(coop_update["picr_loss"], 6),
            "adversary_policy_loss": round(adv_update["policy_loss"], 6),
        }
        train_rows.append(metrics)
        run_log_lines.append(str(metrics))

        if episode_return > best_return:
            best_return = episode_return
            save_npz(out_dir / "best.ckpt", **cooperative_policy.state_dict())
            if adversary_policy is not None:
                save_npz(out_dir / "best_adversary.ckpt", **adversary_policy.state_dict())
        save_npz(out_dir / "latest.ckpt", **cooperative_policy.state_dict())
        if adversary_policy is not None:
            save_npz(out_dir / "latest_adversary.ckpt", **adversary_policy.state_dict())

        episode_index += 1

    permutation_rows, permutation_mean = evaluate_permutation_metric(
        cooperative_policy,
        permutation_obs_samples,
        spec.teammate_slicing,
    )
    to_csv_gz(train_rows, out_dir / "train_metrics.csv.gz")
    to_csv_gz([], out_dir / "eval_crossplay.csv.gz")
    to_csv_gz(permutation_rows, out_dir / "eval_permutation_sensitivity.csv.gz")

    in_team_eval = evaluate_in_team(
        config=config,
        cooperative_policy=cooperative_policy,
        adversary_policy=adversary_policy,
        episodes=config.eval_episodes,
    )

    artifact_mb = round(
        sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file()) / 1024**2,
        4,
    )
    summary = {
        "run_id": run_id,
        "run_tag": config.run_tag,
        "env": config.env,
        "method": config.method,
        "seed": config.seed,
        "profile": config.profile,
        "training_steps": config.total_env_steps,
        "wall_clock_runtime_sec": round(time.time() - start_time, 3),
        "artifact_footprint_mb": artifact_mb,
        "package_versions": package_versions(),
        "git_commit": safe_git_commit(),
        "picr_lambda": config.picr_lambda if method == "picr_ma" else 0.0,
        "teammate_dropout_p": config.teammate_dropout_p,
        "eval_episodes": config.eval_episodes,
        "cooperative_param_count": parameter_count(cooperative_policy),
        "adversary_param_count": parameter_count(adversary_policy) if adversary_policy is not None else 0,
        "train_return": float(np.mean([row["episodic_return"] for row in train_rows[-10:]])),
        "crossplay_return": None,
        "permutation_js": permutation_mean,
        "in_team_eval_return": in_team_eval["return_mean"],
        "collision_mean": in_team_eval["collision_mean"],
        "goal_margin_mean": in_team_eval["goal_margin_mean"],
        "success_mean": in_team_eval["success_mean"],
    }
    write_json(out_dir / "summary.json", summary)
    dump_yaml(asdict(config), out_dir / "config_expanded.yaml")
    write_text(out_dir / "run_log.txt", "\n".join(run_log_lines) + "\n")

    inventory_path = Path("results") / "artifact_inventory.csv"
    existing = []
    if inventory_path.exists():
        existing = inventory_path.read_text(encoding="utf-8").splitlines()
    row = f"{run_id},{artifact_mb}"
    header = "run_id,artifact_mb"
    if not existing:
        inventory_path.write_text(header + "\n" + row + "\n", encoding="utf-8")
    else:
        inventory_path.write_text("\n".join([header] + [line for line in existing[1:] if line.split(",")[0] != run_id] + [row]) + "\n", encoding="utf-8")

    log(f"completed {run_id}")
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="matrix_coordination")
    parser.add_argument("--method", default="ippo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tag", default="manual")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--picr-lambda", type=float, default=0.2)
    parser.add_argument("--teammate-dropout-p", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mpl")))
    args = parse_args()
    run_experiment(
        env=args.env,
        method=args.method,
        seed=args.seed,
        steps=args.steps,
        tag=args.tag,
        max_cycles=args.max_cycles,
        picr_lambda=args.picr_lambda,
        teammate_dropout_p=args.teammate_dropout_p,
        eval_episodes=args.eval_episodes,
    )
