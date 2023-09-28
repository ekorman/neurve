import torch
from neurve.distance import batch_pdist, pdist


def sigmoid_inv(x):
    return torch.clamp(torch.log(x / (1 - x)), -10000, 10000)


def rbf(X, Y, sigma, batch=False):
    d = batch_pdist if batch else pdist
    return torch.exp(-d(X, Y) / 2 / sigma**2)


def imq(X, Y, C, batch=False):
    d = batch_pdist if batch else pdist
    return C / (C + d(X, Y))


def q_loss(q):
    """
    Parameters
    ----------
    q : torch.Tensor
        shape [n, nc] giving chart membership probabilities, where
        n is the batch size, nc the number of charts
    """
    return ((q - 1 / q.shape[1]) ** 2).sum(1).mean()


class MMD:
    def __init__(self, kernel, sigma):
        assert kernel in ["rbf", "imq"]
        k = rbf if kernel == "rbf" else imq
        self.k = lambda X, Y: k(X, Y, sigma)

    def __call__(self, X, Y):
        """Computes the MMD between samples

        x : [N, d]
        """
        n = X.shape[0]
        assert n == Y.shape[0]

        ret = (self.k(X, X).sum() - n) / (n * (n - 1))
        ret += (self.k(Y, Y).sum() - n) / (n * (n - 1))
        ret += -2 * self.k(X, Y).sum() / n**2
        return ret


class MMDManifold:
    def __init__(self, kernel, sigma):
        assert kernel in ["rbf", "imq"]
        k = rbf if kernel == "rbf" else imq
        self.k = lambda X, Y: k(X, Y, sigma, batch=True)

    def __call__(self, q, X, Y):
        """X is the encoder over the data distribution,
        q is the coordinate probabilities for X,
        and Y is sampled from the ambient distribution

        Parameters
        ----------
        q : torch.Tensor
            shape [nc, n] giving chart membership probabilities
        X : torch.Tensor
            shape [nc, n, d] (nc the number of charts, d the ambient dimension)
        Y : torch.Tensor
            shape [n, d] (d the manifold dimension)
        """
        nc, n = X.shape[:2]
        assert n == Y.shape[0]

        Y = Y.unsqueeze(0)

        qq = q.unsqueeze(2) * q.unsqueeze(1)  # shape [nc, N, N]

        ret = (
            self.k(X, X)
            * qq
            * (1 - torch.eye(n, device=q.device).unsqueeze(0))
        ).sum() / (n * (n - 1))
        ret += (self.k(Y, Y).sum() - n) / (nc * n * (n - 1))
        ret += -2 * (q.unsqueeze(2) * self.k(X, Y)).sum() / (nc * n**2)
        return ret


class MMDManifoldLoss:
    def __init__(self, kernel, sigma, device):
        """
        Parameters
        ----------
        kernel : str
            one of "imq" or "rbf"
        sigma : float
        device : str
        """
        self.mmd = MMDManifold(kernel=kernel, sigma=sigma)
        self.device = device

    def __call__(self, q, X):
        """
        Parameters
        ----------
        q : torch.Tensor
            shape [n, nc] giving chart membership probabilities, where
            n is the batch size, nc the number of charts
        X : torch.Tensor
            coordinates. should be shape [n, nc, d] (d the dimension of the
            manifold). Note that X should be the output before the sigmoid activation
            since the MMD will be computed between the sample X and the inverse
            sigmoid of a random sample of the uniform distribution on [0, 1]^d

        Returns
        -------
        torch.Tensor
            the MMD loss (a scalar)
        """
        return self.mmd(
            q.T,
            X.transpose(0, 1),
            sigmoid_inv(
                torch.rand(X.shape[0], X.shape[2], device=self.device)
            ),
        )
