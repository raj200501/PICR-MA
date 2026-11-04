"""Self-play, cross-play and permutation-sensitivity evaluation.

Evaluations reset environments from fixed lists of seeds that training never
uses (``eval_seeds``), so every policy and every pairing is scored on identical
start states (common random numbers). The ``validation`` episodes are used to
select the PICR weight; the ``test`` episodes are used for the reported results.
"""

from __future__ import annotations

import numpy as np
import torch

from picr_ma.envs.mpe import make_env
from picr_ma.envs.slicing import TeammateSlicing, permute_teammates
from picr_ma.networks import ActorCritic

EVAL_SEED_OFFSET = 1_000_000  # training episodes draw environment seeds below this
SPLIT_OFFSETS = {"test": EVAL_SEED_OFFSET, "validation": 2 * EVAL_SEED_OFFSET}


def eval_seeds(n: int, split: str = "test") -> list[int]:
    """``n`` environment seeds reserved for evaluation; disjoint from training and from the other split."""
    return [SPLIT_OFFSETS[split] + i for i in range(n)]


def play_episodes(
    env_name: str,
    team: list[ActorCritic],
    adversary: ActorCritic | list[ActorCritic] | None,
    seeds: list[int],
    sample_seed: int = 0,
    greedy: bool = False,
    collect_obs: bool = False,
    slot_permutations: list[tuple[int, ...]] | None = None,
    resample_teammates: bool = False,
) -> dict[str, np.ndarray]:
    """Roll out one episode per seed (in parallel); team slot ``i`` is controlled by ``team[i]``.

    ``adversary`` may be a list of policies; episodes are assigned to them in
    pairs, episode ``i`` facing ``adversary[(i // 2) % len(adversary)]``, so the
    even and odd episodes see the same opponents (used for split-half checks).

    ``slot_permutations`` gives one teammate-slot permutation per team slot:
    agent ``i`` sees its teammates reordered by ``slot_permutations[i]`` (an
    "order shift": the same state described in a different teammate order).
    ``resample_teammates`` replaces every agent's teammate blocks with those of
    the next episode in the batch at the same step: realistic teammate
    features that carry no information about the agent's own episode.

    Returns per-episode team returns (sum over time of the mean team reward) and
    optionally the visited team observations, flattened to (episodes*T*N, D).
    """
    env = make_env(env_name)
    spec = env.spec
    if len(team) != spec.n_team:
        raise ValueError(f"{env_name} has {spec.n_team} team slots, got {len(team)} policies")
    adversaries = list(adversary) if isinstance(adversary, (list, tuple)) else [adversary]
    if spec.n_adversaries and any(a is None for a in adversaries):
        raise ValueError(f"{env_name} needs an adversary policy")
    assignment = (np.arange(len(seeds)) // 2) % len(adversaries)
    blocks = list(np.ravel(spec.slicing.teammate_blocks))
    generator = torch.Generator().manual_seed(sample_seed)
    obs = env.reset(seeds)
    returns = np.zeros(len(seeds))
    visited = []
    done = False
    while not done:
        team_obs = obs["team"]
        if resample_teammates:
            team_obs = team_obs.copy()
            team_obs[:, :, blocks] = np.roll(obs["team"], 1, axis=0)[:, :, blocks]
        if slot_permutations is not None:
            team_obs = np.stack(
                [
                    permute_teammates(team_obs[:, slot], spec.slicing, np.asarray(perm))
                    for slot, perm in enumerate(slot_permutations)
                ],
                axis=1,
            )
        if collect_obs:
            visited.append(team_obs.reshape(-1, spec.obs_dim))
        actions = np.stack([policy.act(team_obs[:, slot], generator, greedy)[0] for slot, policy in enumerate(team)], axis=1)
        adv_actions = None
        if spec.n_adversaries:
            adv_actions = np.empty((len(seeds), spec.n_adversaries), dtype=np.int64)
            for k, policy in enumerate(adversaries):
                rows = assignment == k
                if not rows.any():
                    continue
                adv_obs = obs["adversary"][rows].reshape(-1, spec.adversary_obs_dim)
                adv_actions[rows] = policy.act(adv_obs, generator, greedy)[0].reshape(-1, spec.n_adversaries)
        obs, team_rewards, _, done = env.step(actions, adv_actions)
        returns += team_rewards.mean(axis=1)
    out = {"returns": returns}
    if collect_obs:
        out["obs"] = np.concatenate(visited, axis=0)
    return out


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence (natural log, bounded by ln 2)."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


def permutation_sensitivity(policy: ActorCritic, obs: np.ndarray, slicing: TeammateSlicing) -> dict[str, float]:
    """How much the policy's output depends on teammate order.

    For each observation, compares the action distribution on the original
    observation with the distribution under every non-identity teammate
    permutation. Reports the mean JS divergence and the fraction of
    (observation, permutation) pairs whose greedy action changes.
    """
    perms = slicing.all_permutations(include_identity=False)
    if not perms or len(obs) == 0:
        return {"perm_js": 0.0, "perm_flip_rate": 0.0}
    base = policy.action_probs(obs)
    js, flips = [], []
    for perm in perms:
        permuted = policy.action_probs(permute_teammates(obs, slicing, np.asarray(perm)))
        js.append(js_divergence(base, permuted))
        flips.append(base.argmax(-1) != permuted.argmax(-1))
    return {"perm_js": float(np.mean(js)), "perm_flip_rate": float(np.mean(flips))}


def evaluate_self_play(env_name: str, policy: ActorCritic, adversary: ActorCritic | None, episodes: int) -> dict[str, float]:
    spec = make_env(env_name).spec
    result = play_episodes(env_name, [policy] * spec.n_team, adversary, eval_seeds(episodes), collect_obs=True)
    return {
        "self_play_return": float(result["returns"].mean()),
        "self_play_return_std": float(result["returns"].std()),
        **permutation_sensitivity(policy, result["obs"], spec.slicing),
    }
