import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset


from neurve.distance import pdist as torch_pdist
from neurve.mmd import MMDManifoldLoss


class LaplacianEigenvalueLoss(nn.Module):
    def __init__(self, lam=0.1):
        self.lam = lam

    def forward(self, x):
        dists = torch_pdist(x, x)
        ret = (g * dists).sum() / E.shape[0]

        if self.lam != 0:
            ret += self.lam * torch.abs((x ** 2).sum(1).mean() - 1)

        return ret


def lap_eig_obj(g, E, lam=0.1):
    """
    Parameters
    ----------
    g
        nn graph
    E
        embeddings
    """
    dists = torch_pdist(E, E)
    return (g * dists).sum() / E.shape[
        0
    ]  # + lam * torch.abs((E ** 2).sum(1).mean() - 1)


def nearest_neighbors(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Parameters
    ----------
    X
        array of shape [N, d] of N many d-dimensional vectors
    n_neighbors
        number of nearest neighbors to find

    Returns
    -------
    np.ndarray
        shape [N, n_neighbors] giving the indices of the nearest neighbors.
        Note that this may not be sorted, e.g. the index at [i, 0] may not be
        the closest neighbor to X[i] but it is among the closest
        n_neighbors many.
    """
    dists = squareform(pdist(X))
    for i in range(dists.shape[0]):
        dists[i, i] = np.infty

    ret = np.argpartition(dists, n_neighbors, 1)
    return ret[:, :n_neighbors]


def nn_graph(X, n_neighbors) -> np.ndarray:
    N = X.shape[0]
    nn = nearest_neighbors(X, n_neighbors)
    row_ind = [i for i in range(N) for _ in range(n_neighbors)]
    col_ind = [j for i in range(N) for j in nn[i]]
    data = [1] * N * n_neighbors

    ret = np.zeros((N, N))
    ret[row_ind, col_ind] = data
    return ret


def main(X: np.ndarray, dim_z: int, n_charts: int):
    N = X.shape[0]
    q = torch.rand(size=(N, n_charts), requires_grad=True)
    # embeddings
    E = torch.rand(size=(N, n_charts, dim_z), requires_grad=True)


def lap_eig_maps(X: np.ndarray, dim_z: int, n_neighbors, opt_steps=1000):
    """Trains against the Laplacian eigenmap objective
    """
    N = X.shape[0]
    g = torch.tensor(nn_graph(X, n_neighbors))
    E = torch.rand(size=(N, dim_z), requires_grad=True)

    opt = SGD(params=[E], lr=10)
    for i in range(opt_steps):
        loss = lap_eig_obj(g, E)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 1:
            print(f"Loss: {loss}")

    return E.detach().numpy()


# X = np.random.uniform(0, 1, size=(5, 10))
X, color = datasets.make_s_curve(1000)

E = lap_eig_maps(X, 2, 10, 500)

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

ax = fig.add_subplot(2, 1, 2)
ax.scatter(E[:, 0], E[:, 1], c=color, cmap=plt.cm.Spectral)

plt.show()

g = nn_graph(X, 3)
