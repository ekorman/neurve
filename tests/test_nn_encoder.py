import numpy as np
import pytest
import torch

from neurve.nn_encoder.dataset import NNDataset
from neurve.nn_encoder.loss import loss, loss_at_a_point
from neurve.nn_encoder.models import MfldEncoder
from neurve.nn_encoder.trainer import NNEncoderTrainer

n = 8
z = 2
batch = 2
n_charts = 3
c = 0.1


@pytest.fixture
def net() -> MfldEncoder:
    return MfldEncoder(
        n=n,
        z=z,
        backbone_dim=4,
        hidden_dim=6,
        n_charts=n_charts,
        use_batch_norm=False,
    )


@pytest.fixture
def dataset() -> NNDataset:
    return NNDataset(data=np.random.rand(15, n), n_neighbors=4)


def test_mfld_encoder(net: MfldEncoder):
    x = torch.rand(batch, n)
    q, coords = net(x)

    assert q.shape == (batch, n_charts)
    assert coords.shape == (batch, n_charts, z)

    assert q.max() <= 1
    assert q.min() >= 0
    assert q.sum(1).allclose(torch.ones(batch))


def test_loss():
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


def test_dataset(dataset: NNDataset):
    assert len(dataset) == 15
    pt, nbrs, non_nbrs = dataset[2]
    assert pt.dtype == np.float32
    assert np.array_equal(pt, dataset.data[2].astype(np.float32))
    assert nbrs.shape == (4, n)
    assert non_nbrs.shape == (4, n)

    nbr_dists = np.linalg.norm(nbrs - pt, axis=1)
    non_nbr_dists = np.linalg.norm(non_nbrs - pt, axis=1)

    assert nbr_dists.max() < non_nbr_dists.min()


def test_trainer(net: MfldEncoder, dataset: NNDataset):
    batch_size = 2
    opt = torch.optim.Adam(net.parameters())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True
    )

    inital_params = [p.detach().clone() for p in net.parameters()]

    trainer = NNEncoderTrainer(
        net=net,
        opt=opt,
        out_path="out",
        reg_loss_weight=0.4,
        data_loader=data_loader,
        c=c,
    )

    trainer.train(n_epochs=2)

    assert trainer.global_steps == 2 * (len(dataset.data) // batch_size)

    final_weights = list(net.parameters())

    # check weights changed
    a_weight_changed = False
    for init, final in zip(inital_params, final_weights):
        if not torch.allclose(init, final):
            a_weight_changed = True
            break
    assert a_weight_changed
