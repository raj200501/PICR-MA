"""Train one (environment, method, seed) run and write its artifacts.

Methods (all share the same PPO learner and hyperparameters):

* ``ippo``      - shared-parameter IPPO with an MLP policy.
* ``picr``      - IPPO + the PICR consistency penalty (training-time only).
* ``deepsets``  - IPPO with a teammate-permutation-invariant Deep Sets encoder.
* ``order_aug`` - IPPO where each policy input has its teammate blocks randomly
                  permuted (data augmentation over the observation symmetry).

Each PPO update uses ``num_envs`` complete episodes collected in parallel.
Artifacts written to ``out_dir``: config.json, metrics.csv.gz (one row per
logged update), summary.json (final evaluation), policy.pt and, for
environments with an adversary, adversary.pt.
"""

from __future__ import annotations

import csv
import gzip
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from picr_ma.envs.mpe import make_env
from picr_ma.envs.slicing import permute_teammates, random_permutations
from picr_ma.evaluation import EVAL_SEED_OFFSET, evaluate_self_play, permutation_sensitivity
from picr_ma.networks import ActorCritic, save_policy
from picr_ma.ppo import PPOConfig, compute_gae, ppo_update

METHODS = ("ippo", "picr", "deepsets", "order_aug")


@dataclass
class TrainConfig:
    env: str
    method: str
    seed: int
    total_steps: int = 3_000_000
    num_envs: int = 64
    hidden: int = 64
    picr_lambda: float = 0.1
    eval_episodes: int = 500
    log_every: int = 5
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise KeyError(f"Unknown method {self.method!r}; expected one of {METHODS}")
        if isinstance(self.ppo, dict):
            self.ppo = PPOConfig(**self.ppo)


def _policy_input(method: str, obs: np.ndarray, slicing, rng: np.random.Generator) -> np.ndarray:
    if method == "order_aug" and slicing.n_teammates > 1:
        return permute_teammates(obs, slicing, random_permutations(len(obs), slicing.n_teammates, rng))
    return obs


def _batch(steps: dict[str, list], gamma: float, gae_lambda: float) -> dict[str, np.ndarray]:
    """Stack per-step (B, agents, ...) arrays into (B, T, agents, ...) and flatten for PPO."""
    arrays = {k: np.stack(v, axis=1) for k, v in steps.items()}
    advantages, returns = compute_gae(arrays["rewards"], arrays["values"], gamma, gae_lambda)
    flat = lambda x: x.reshape(-1, *x.shape[3:])  # noqa: E731
    return {
        "obs": flat(arrays["obs"]),
        "actions": flat(arrays["actions"]),
        "log_probs": flat(arrays["log_probs"]),
        "advantages": flat(advantages),
        "returns": flat(returns),
    }


def _act(model: ActorCritic, obs: np.ndarray, generator: torch.Generator):
    """Act on (B, agents, D) observations with a shared policy."""
    b, n, d = obs.shape
    actions, log_probs, values = model.act(obs.reshape(b * n, d), generator)
    return actions.reshape(b, n), log_probs.reshape(b, n), values.reshape(b, n)


def train(config: TrainConfig, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng([config.seed, 0])
    env_rng = np.random.default_rng([config.seed, 1])
    generator = torch.Generator().manual_seed(config.seed)

    env = make_env(config.env)
    spec = env.spec
    policy = ActorCritic(
        spec.obs_dim,
        spec.n_actions,
        encoder="deepsets" if config.method == "deepsets" else "mlp",
        hidden=config.hidden,
        slicing=spec.slicing,
    )
    team_ppo = PPOConfig(**{**asdict(config.ppo), "picr_lambda": config.picr_lambda if config.method == "picr" else 0.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=team_ppo.lr, eps=1e-5)
    adversary = adv_optimizer = None
    if spec.n_adversaries:
        adversary = ActorCritic(spec.adversary_obs_dim, spec.n_actions, hidden=config.hidden)
        adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=config.ppo.lr, eps=1e-5)
    adv_ppo = PPOConfig(**{**asdict(config.ppo), "picr_lambda": 0.0})

    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    metrics_file = gzip.open(out_dir / "metrics.csv.gz", "wt", newline="")
    writer = None
    env_steps = update = 0
    update_seconds = 0.0
    start = time.perf_counter()

    while env_steps < config.total_steps:
        obs = env.reset(env_rng.integers(EVAL_SEED_OFFSET, size=config.num_envs))
        team = {k: [] for k in ("obs", "actions", "log_probs", "values", "rewards")}
        adv = {k: [] for k in team}
        episode_return = np.zeros(config.num_envs)
        done = False
        while not done:
            b, n, d = obs["team"].shape
            policy_obs = _policy_input(config.method, obs["team"].reshape(b * n, d), spec.slicing, rng).reshape(b, n, d)
            actions, log_probs, values = _act(policy, policy_obs, generator)
            adv_actions = None
            if adversary is not None:
                adv_obs = obs["adversary"]
                adv_actions, adv_log_probs, adv_values = _act(adversary, adv_obs, generator)
            obs, team_rewards, adv_rewards, done = env.step(actions, adv_actions)
            for key, value in zip(team, (policy_obs, actions, log_probs, values, team_rewards)):
                team[key].append(value)
            if adversary is not None:
                for key, value in zip(adv, (adv_obs, adv_actions, adv_log_probs, adv_values, adv_rewards)):
                    adv[key].append(value)
            episode_return += team_rewards.mean(axis=1)
        env_steps += env.horizon * config.num_envs
        update += 1

        update_start = time.perf_counter()
        batch = _batch(team, team_ppo.gamma, team_ppo.gae_lambda)
        stats = ppo_update(policy, optimizer, batch, team_ppo, rng, spec.slicing)
        if adversary is not None:
            ppo_update(adversary, adv_optimizer, _batch(adv, adv_ppo.gamma, adv_ppo.gae_lambda), adv_ppo, rng)
        update_seconds += time.perf_counter() - update_start

        if update % config.log_every == 0 or env_steps >= config.total_steps:
            # Mean over all non-identity permutations, so it is the same whether
            # measured on raw or (for order_aug) permuted inputs.
            row = {
                "env_steps": env_steps,
                "episode_return": float(episode_return.mean()),
                **stats,
                **permutation_sensitivity(policy, batch["obs"][:2048], spec.slicing),
            }
            row = {k: v if isinstance(v, int) else float(f"{v:.5g}") for k, v in row.items()}
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    metrics_file.close()
    train_seconds = time.perf_counter() - start

    save_policy(policy, out_dir / "policy.pt", spec.slicing)
    if adversary is not None:
        save_policy(adversary, out_dir / "adversary.pt")
    policy.eval()
    summary = {
        "env": config.env,
        "method": config.method,
        "seed": config.seed,
        "picr_lambda": team_ppo.picr_lambda,
        "env_steps": env_steps,
        "train_seconds": round(train_seconds, 2),
        "update_seconds": round(update_seconds, 2),
        "n_params": sum(p.numel() for p in policy.parameters()),
        **evaluate_self_play(config.env, policy, adversary, config.eval_episodes),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
