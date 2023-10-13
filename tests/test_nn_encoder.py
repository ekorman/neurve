import numpy as np
import torch

from neurve.nn_encoder.dataset import NNDataset
from neurve.nn_encoder.loss import loss, loss_at_a_point
from neurve.nn_encoder.models import MfldEncoder


def test_mfld_encoder():
    n = 8
    z = 2
    batch = 2
    n_charts = 3

    net = MfldEncoder(
        n=n,
        z=z,
        backbone_dim=4,
        hidden_dim=6,
        n_charts=n_charts,
        use_batch_norm=False,
    )

    x = torch.rand(batch, n)
    q, coords = net(x)

    assert q.shape == (batch, n_charts)
    assert coords.shape == (batch, n_charts, z)

    assert q.max() <= 1
    assert q.min() >= 0
    assert q.sum(1).allclose(torch.ones(batch))


def test_loss():
    n = 8
    z = 2
    n_charts = 3
    n_neighbors = 4
    n_non_neighbors = 5

    point = torch.rand(n)
    neighbors = torch.rand(n_neighbors, n)
    q_point = torch.rand(n_charts)
    q_neighbors = torch.rand(n_neighbors, n_charts)
    q_non_neighbors = torch.rand(n_non_neighbors, n_charts)
    coords_point = torch.rand(n_charts, z)
    coords_neighbors = torch.rand(n_neighbors, n_charts, z)
    coords_non_neighbors = torch.rand(n_non_neighbors, n_charts, z)
    scale = torch.Tensor([0.75])
    c = 0.1

    loss_dict = loss_at_a_point(
        point=point,
        neighbors=neighbors,
        q_point=q_point,
        q_neighbors=q_neighbors,
        q_non_neighbors=q_non_neighbors,
        coords_point=coords_point,
        coords_neighbors=coords_neighbors,
        coords_non_neighbors=coords_non_neighbors,
        scale=scale,
        c=c,
    )

    assert set(loss_dict.keys()) == {
        "loss_neighbors",
        "loss_non_neighbors",
        "loss",
    }

    def _double(x):
        return torch.stack([x, x])

    batch_loss_dict = loss(
        point=_double(point),
        neighbors=_double(neighbors),
        q_point=_double(q_point),
        q_neighbors=_double(q_neighbors),
        q_non_neighbors=_double(q_non_neighbors),
        coords_point=_double(coords_point),
        coords_neighbors=_double(coords_neighbors),
        coords_non_neighbors=_double(coords_non_neighbors),
        scale=scale,
        c=c,
    )

    for k, v in loss_dict.items():
        assert v.allclose(batch_loss_dict[k])


def test_dataset():
    data = np.random.rand(15, 2)
    dset = NNDataset(data=data, n_neighbors=4)

    assert len(dset) == 15
    pt, nbrs, non_nbrs = dset[2]
    assert np.array_equal(pt, data[2])
    assert nbrs.shape == (4, 2)
    assert non_nbrs.shape == (4, 2)

    nbr_dists = np.linalg.norm(nbrs - pt, axis=1)
    non_nbr_dists = np.linalg.norm(non_nbrs - pt, axis=1)

    assert nbr_dists.max() < non_nbr_dists.min()
