import numpy as np
import torch

from picr_ma.envs.slicing import TeammateSlicing
from picr_ma.networks import ActorCritic
from picr_ma.regularizers import picr_loss, symmetric_kl

SLICING = TeammateSlicing(obs_dim=10, teammate_blocks=((2, 3, 4), (5, 6, 7)))


def test_symmetric_kl_matches_definition():
    a, b = torch.randn(32, 5), torch.randn(32, 5)
    p, q = torch.softmax(a, -1), torch.softmax(b, -1)
    expected = 0.5 * ((p * (p / q).log()).sum(-1) + (q * (q / p).log()).sum(-1)).mean()
    torch.testing.assert_close(symmetric_kl(a, b), expected)
    torch.testing.assert_close(symmetric_kl(a, b), symmetric_kl(b, a))
    assert symmetric_kl(a, a).item() == 0.0


def test_symmetric_kl_gradient():
    a = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(symmetric_kl, (a, b))


def test_picr_loss_is_zero_for_invariant_policy():
    obs = torch.randn(128, 10)
    deepsets = ActorCritic(10, 5, encoder="deepsets", slicing=SLICING)
    assert picr_loss(deepsets.logits, obs, SLICING, np.random.default_rng(0)).item() < 1e-10


def test_minimizing_picr_loss_makes_mlp_invariant():
    torch.manual_seed(0)
    obs = torch.randn(512, 10)
    model = ActorCritic(10, 5)
    model.policy_head.weight.data.normal_()
    rng = np.random.default_rng(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = picr_loss(model.logits, obs, SLICING, np.random.default_rng(1)).item()
    for _ in range(300):
        loss = picr_loss(model.logits, obs, SLICING, rng)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = picr_loss(model.logits, obs, SLICING, np.random.default_rng(1)).item()
    assert initial > 1e-2
    assert final < 0.02 * initial
