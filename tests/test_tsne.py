import math

import pytest
import torch

from neurve.dim_red.tsne import cond_matrix, find_var2, perplexity
from neurve.distance import pdist


def test_cond_matrix():
    N = 5
    X = torch.rand((N, 4))
    var2 = torch.rand(N)

    cond = cond_matrix(var2=var2, X=X)
    assert cond.shape == torch.Size((N, N))

    for i in range(N):
        for j in range(N):
            num = math.exp(-((X[i] - X[j]) ** 2).sum() / (2 * var2[i]))
            denom = 0
            for k in range(N):
                if k == i:
                    continue
                denom += math.exp(-((X[i] - X[k]) ** 2).sum() / (2 * var2[i]))

            assert cond[i, j].item() == pytest.approx(num / denom, abs=1e-3)


def test_find_var2():
    N, d = 50, 64
    target_perp = 15.0
    tol = 1e-4
    X = torch.rand((N, d))
    dists = pdist(X, X)
    var2 = find_var2(0, 100, target_perp, tol, dists)

    cond = cond_matrix(var2=var2, dists=dists)
    p = perplexity(cond)

    torch.testing.assert_allclose(p, target_perp)
