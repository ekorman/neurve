import torch
from neurve.contrastive.loss import SimCLRLoss
from neurve.distance import psim


def test_simclr_loss():
    """Test that SimCLRLoss agrees with the paper
    https://arxiv.org/abs/2002.05709"""
    N, d, tau = 3, 12, 0.1
    z1 = torch.rand(N, d)  # odds
    z2 = torch.rand(N, d)  # evens
    z = torch.stack([z1[0], z2[0], z1[1], z2[1], z1[2], z2[2]])
    s = psim(z, z)

    sim_clr_loss = SimCLRLoss(tau=tau)

    def ell(i, j):
        numerator = torch.exp(s[i, j] / tau)
        denominator = 0
        for k in range(2 * N):
            if k != i:
                denominator += torch.exp(s[i, k] / tau)
        return -torch.log(numerator / denominator)

    loss = 0
    for k in range(N):
        loss += ell(2 * k + 1, 2 * k) + ell(2 * k, 2 * k + 1)
    loss /= 2 * N

    torch.testing.assert_close(loss, sim_clr_loss(z1, z2))
