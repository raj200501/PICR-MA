"""The PICR consistency penalty.

For a batch of observations, sample two independent random permutations of the
teammate blocks per row and penalize the symmetric KL divergence between the
policy's action distributions on the two views:

    L_PICR = E_obs E_{pi1, pi2} [ 0.5 KL(p(.|pi1 o) || p(.|pi2 o)) + 0.5 KL(p(.|pi2 o) || p(.|pi1 o)) ]

Gradients flow through both views. The penalty is zero exactly when the policy
is invariant to teammate order on the sampled observations.
"""

from __future__ import annotations

import numpy as np
import torch

from picr_ma.envs.slicing import TeammateSlicing, permute_teammates, random_permutations


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of 0.5 * (KL(a||b) + KL(b||a)) between categorical distributions."""
    log_a = torch.log_softmax(logits_a, dim=-1)
    log_b = torch.log_softmax(logits_b, dim=-1)
    kl_ab = (log_a.exp() * (log_a - log_b)).sum(-1)
    kl_ba = (log_b.exp() * (log_b - log_a)).sum(-1)
    return 0.5 * (kl_ab + kl_ba).mean()


def picr_views(obs: torch.Tensor, slicing: TeammateSlicing, rng: np.random.Generator):
    n = obs.shape[0]
    first = permute_teammates(obs, slicing, random_permutations(n, slicing.n_teammates, rng))
    second = permute_teammates(obs, slicing, random_permutations(n, slicing.n_teammates, rng))
    return first, second


def picr_loss(policy_logits, obs: torch.Tensor, slicing: TeammateSlicing, rng: np.random.Generator) -> torch.Tensor:
    """PICR penalty for ``policy_logits`` (a callable obs -> logits) on ``obs``."""
    if slicing.n_teammates < 2:
        return obs.new_zeros(())
    first, second = picr_views(obs, slicing, rng)
    return symmetric_kl(policy_logits(first), policy_logits(second))
