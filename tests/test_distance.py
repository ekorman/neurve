import torch
from neurve.distance import distmfld, pdist_mfld, psim


def test_distmfld_disjoint():
    """Test that points that are not in a common chart have distance 1"""
    q1 = torch.tensor([[1], [0]])
    q2 = torch.tensor([[0], [1]])
    c1, c2 = [torch.rand(2, 1, 3) for _ in range(2)]

    assert distmfld(q1, c1, q2, c2) == 1.0


def test_pdist_mfld():
    """Test batch computation of pairwise distances"""
    nc, m, n, d = 4, 5, 6, 7
    q1 = torch.softmax(torch.rand(nc, m), 0)
    q2 = torch.softmax(torch.rand(nc, n), 0)
    c1 = torch.rand(nc, m, d)
    c2 = torch.rand(nc, n, d)

    pdist = pdist_mfld(q1, c1, q2, c2)
    for i in range(m):
        for j in range(n):
            torch.testing.assert_close(
                pdist[i, j],
                distmfld(
                    q1[:, i : i + 1],
                    c1[:, i : i + 1],
                    q2[:, j : j + 1],
                    c2[:, j : j + 1],
                )[0],
            )


def test_psim():
    """Test batch computation of pairwise cosine similarities"""
    m, n, d = 10, 12, 7
    X, Y = torch.rand(m, d), torch.rand(n, d)
    sims = psim(X, Y)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            torch.testing.assert_close(
                torch.dot(x, y) / (torch.norm(x) * torch.norm(y)), sims[i, j]
            )
