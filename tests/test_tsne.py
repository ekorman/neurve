import pytest
import torch

from neurve.tsne.stats import cond_matrix, get_cond_dist_given_perp, joint_q, perplexity
from neurve.distance import pdist


def test_cond_matrix():
    N = 5
    X = torch.rand((N, 4))
    var2 = torch.rand(N)

    cond = cond_matrix(var2=var2, X=X)
    assert cond.shape == torch.Size((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                assert cond[i, j].item() == pytest.approx(0, abs=1e-9)
                continue
            num = torch.exp(-((X[i] - X[j]) ** 2).sum() / (2 * var2[i]))
            denom = 0
            for k in range(N):
                if k == i:
                    continue
                denom += torch.exp(-((X[i] - X[k]) ** 2).sum() / (2 * var2[i]))

            assert cond[i, j].item() == pytest.approx((num / denom).item(), abs=1e-3)


def test_joint_q():
    N = 5
    E = torch.rand((N, 4))

    t_dist = joint_q(E)
    for i in range(N):
        for j in range(N):
            if i == j:
                assert t_dist[i, j].item() == 0
                continue
            num = 1 / (1 + ((E[i] - E[j]) ** 2).sum())
            denom = 0
            for k in range(N):
                for l in range(N):
                    if k == l:
                        continue
                    denom += 1 / (1 + ((E[k] - E[l]) ** 2).sum())
            assert t_dist[i, j].item() == pytest.approx((num / denom).item(), abs=1e-6)


def test_get_cond_dist_given_perp():
    N, d = 50, 64
    target_perp = 15.0
    tol = 1e-4
    X = torch.rand((N, d))
    cond = get_cond_dist_given_perp(0, 100, target_perp, tol, X)

    p = perplexity(cond)

    torch.testing.assert_allclose(p, target_perp)
