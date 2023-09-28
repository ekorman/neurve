import torch
from neurve.contrastive.models import SimCLRMfld
from neurve.core.models import CoordLinear


def test_simclrmfld():
    """Test that outputs of SimCLRMfld have the correct shapes"""
    dim_z, n_charts, batch_size, proj_dim = 4, 7, 5, 20
    net = SimCLRMfld(dim_z=dim_z, n_charts=n_charts, proj_dim=proj_dim)

    x = torch.rand(batch_size, 3, 16, 18)

    q, coords = net.encode(x)
    proj_head = net(x)

    assert q.shape == (batch_size, n_charts)
    assert coords.shape == (batch_size, n_charts, dim_z)
    assert proj_head.shape == (batch_size, proj_dim)


def test_coord_linear():
    """Test that the coordinate-linear model has the proper functionality"""
    dim_z, n_charts, out_dim, batch_size = 5, 7, 6, 12

    coords = torch.rand(batch_size, n_charts, dim_z)
    q = torch.rand(batch_size, n_charts)

    net = CoordLinear(dim_z, n_charts, out_dim, one_hot_q=False)

    out = net(q, coords)
    coords = torch.sigmoid(coords)
    assert out.shape == (batch_size, out_dim)

    for i in range(batch_size):
        out_i = 0
        for j in range(n_charts):
            out_i += q[i, j] * (net.weights[j] @ coords[i, j] + net.biases[j])

        torch.testing.assert_close(out_i, out[i])
