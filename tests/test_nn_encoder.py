import torch

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
