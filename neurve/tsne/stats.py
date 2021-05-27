import torch

from neurve.distance import pdist


def cond_matrix(var2, X=None, dists=None):
    """
    Parameters
    ----------
    X : torch.Tensor
        shape [N, d]
    dists : torch.Tensor
        shape [M, N]
    var : torch.Tensor
        shape [N]

    Returns
    -------
    torch.Tensor
        has shape [N, N]. The entry at [i, j] is the
        conditional probability that x_i would pick x_j as its neighbor.
        (Equation (1) in "Visualizing Data using t-SNE")
    """
    assert (X is None) != (dists is None)
    if X is not None:
        dists = pdist(X, X)
    inv_var2 = (1 / var2).diag()
    exps = torch.exp(-inv_var2 @ dists / 2)
    exps.fill_diagonal_(0.0)
    denom = exps.sum(dim=1, keepdim=True)
    ret = exps / denom
    return ret + 1e-9


def joint_q(E):
    dists = pdist(E, E)
    t = 1 / (1 + dists)
    t.fill_diagonal_(0.0)
    denom = t.sum()
    return t / denom


def entropy(P):
    """
    Parameters
    ----------
    P : torch.Tensor
        shape [M, N]
    """
    return -(P * P.log2()).sum(1)


def perplexity(p):
    return 2 ** entropy(p)


def kl_div(P, Q):
    """
    Parameters
    ----------
    P : torch.Tensor
        shape [N, N]
    Q : torch.Tensor
        shape [N, N]

    Returns
    -------
    torch.Tensor
        scalar
    """
    return (P * torch.log(P / Q)).sum()


def find_var2(min_val, max_val, target, tol, dists, max_iters=1000):
    """
    Parameters
    ----------
    min_val : float
    max_val : float
    target : float
    tol : float
    dists: torch.Tensor
        shape [N, N]
    """
    var2 = None
    N = dists.shape[0]
    finished = torch.zeros(N, dtype=bool)
    min_val = min_val * torch.ones(N)
    max_val = max_val * torch.ones(N)

    for _ in range(max_iters):
        if var2 is None:
            var2 = 0.5 * (min_val + max_val)
        else:
            var2 = 0.5 * (min_val + max_val) * (~finished) + var2 * finished
        cond_dist = cond_matrix(var2, dists=dists)
        perp = perplexity(cond_dist)
        diff = perp - target

        finished = torch.abs(diff) < tol

        if finished.all():
            return var2

        max_val = (diff > 0) * var2 + ~(diff > 0) * max_val
        min_val = ~(diff > 0) * var2 + (diff > 0) * min_val

    raise RuntimeError("maximum iterations reached")