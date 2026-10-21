import numpy as np
import torch

from picr_ma.networks import ActorCritic
from picr_ma.ppo import PPOConfig, compute_gae, ppo_update


def _reference_gae(rewards, values, gamma, lam):
    """Textbook single-trajectory GAE, applied to each (episode, agent) separately."""
    adv = np.zeros_like(rewards)
    for e in range(rewards.shape[0]):
        for i in range(rewards.shape[2]):
            running = 0.0
            for t in reversed(range(rewards.shape[1])):
                next_v = values[e, t + 1, i] if t + 1 < rewards.shape[1] else 0.0
                delta = rewards[e, t, i] + gamma * next_v - values[e, t, i]
                running = delta + gamma * lam * running
                adv[e, t, i] = running
    return adv


def test_gae_matches_reference_and_agents_are_independent():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(3, 7, 4)).astype(np.float32)
    values = rng.normal(size=(3, 7, 4)).astype(np.float32)
    adv, ret = compute_gae(rewards, values, 0.97, 0.9)
    np.testing.assert_allclose(adv, _reference_gae(rewards, values, 0.97, 0.9), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, adv + values)
    values2 = values.copy()
    values2[:, :, 1] += 100.0  # changing agent 1's critic must not affect agent 0
    adv2, _ = compute_gae(rewards, values2, 0.97, 0.9)
    np.testing.assert_array_equal(adv[:, :, 0], adv2[:, :, 0])


def test_ppo_learns_contextual_bandit():
    """Regression test: the policy must move *toward* high-advantage actions."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ActorCritic(4, 3, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    config = PPOConfig(epochs=4, minibatches=2, ent_coef=0.0)
    generator = torch.Generator().manual_seed(0)
    contexts = np.eye(4, dtype=np.float32)[:2]  # two contexts with different best actions
    best = np.array([2, 0])
    for _ in range(60):
        ctx = rng.integers(0, 2, size=256)
        obs = contexts[ctx]
        actions, log_probs, values = model.act(obs, generator)
        rewards = (actions == best[ctx]).astype(np.float32)
        batch = {"obs": obs, "actions": actions, "log_probs": log_probs, "advantages": rewards - values, "returns": rewards}
        ppo_update(model, optimizer, batch, config, rng)
    probs = model.action_probs(contexts)
    assert probs[0, 2] > 0.9 and probs[1, 0] > 0.9
