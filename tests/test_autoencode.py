import torch

from neurve.autoencode.model import VectorAutoencoder


def test_vector_autoencoder():
    n, z, backbone_dim, hidden_dim, n_charts = 10, 2, 3, 4, 5

    model = VectorAutoencoder(
        n=n,
        z=z,
        backbone_dim=backbone_dim,
        hidden_dim=hidden_dim,
        n_charts=n_charts,
        use_batch_norm=False,
    )
    x = torch.randn(6, n)
    y = model(x)
    assert y.shape == (6, n)

    q, coords = model.encoder(x)
    assert q.shape == (6, n_charts)
    assert coords.shape == (6, n_charts, z)
