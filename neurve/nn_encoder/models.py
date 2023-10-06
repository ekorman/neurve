import torch
import torch.nn as nn


def create_basic_layer(
    in_dim: int, out_dim: int, use_batch_norm: bool
) -> nn.Module:
    layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


class MfldEncoder(nn.Module):
    def __init__(
        self,
        n: int,
        z: int,
        backbone_dim: int,
        hidden_dim: int,
        n_charts: int,
        use_batch_norm: bool,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            create_basic_layer(n, hidden_dim, use_batch_norm),
            create_basic_layer(hidden_dim, backbone_dim, use_batch_norm),
        )

        self.q = nn.Sequential(
            create_basic_layer(
                in_dim=backbone_dim,
                out_dim=n_charts,
                use_batch_norm=use_batch_norm,
            ),
            nn.Softmax(1),
        )

        self.coord_maps = [
            create_basic_layer(
                in_dim=backbone_dim,
                out_dim=z,
                use_batch_norm=use_batch_norm,
            )
            for _ in range(n_charts)
        ]

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """encodes the input tensor x

        Returns
        -------
        first tensor returned is the chart membership probabilities, shape (batch, n_charts)
        second tensor returned is the coordinates in each chart, shape (batch, n_charts, z)
        """
        x = self.backbone(x)
        coords = [c(x) for c in self.coord_maps]
        coords = torch.stack(coords, 1)
        q = self.q(x)

        return q, coords
