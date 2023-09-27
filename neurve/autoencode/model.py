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

        self.q = create_basic_layer(
            in_dim=backbone_dim,
            out_dim=n_charts,
            use_batch_norm=use_batch_norm,
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
        """encodes the input tensor x"""
        x = self.backbone(x)
        coords = torch.stack([c(x) for c in self.coord_maps], 1)
        q = self.q(x)

        return q, coords


class MfldDecoder(nn.Module):
    def __init__(
        self,
        n: int,
        z: int,
        backbone_dim: int,
        n_charts: int,
        use_batch_norm: bool,
        hidden_dim: int,
    ):
        super().__init__()

        self.qT = create_basic_layer(
            in_dim=n_charts,
            out_dim=backbone_dim,
            use_batch_norm=use_batch_norm,
        )
        self.coordT_maps = [
            create_basic_layer(
                in_dim=z,
                out_dim=backbone_dim,
                use_batch_norm=use_batch_norm,
            )
            for _ in range(n_charts)
        ]
        self.backboneT = nn.Sequential(
            create_basic_layer(backbone_dim, hidden_dim, use_batch_norm),
            create_basic_layer(hidden_dim, n, use_batch_norm),
        )

    def forward(self, q: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """decodes from atlas representation to original space

        Parameters
        ----------
        q
            shape (batch_size, n_charts)
        coords
            shape (batch_size, n_charts, z)

        Returns
        -------
        tensor of shape (batch_size, in_dim)
        """
        x = self.qT(q) + torch.stack(
            [c(coords[:, i]) for i, c in enumerate(self.coordT_maps)]
        ).sum(0)
        x = self.backboneT(x)
        return x


class VectorAutoencoder(nn.Module):
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
        self.encoder = MfldEncoder(
            n=n,
            z=z,
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            n_charts=n_charts,
            use_batch_norm=use_batch_norm,
        )
        self.decoder = MfldDecoder(
            n=n,
            z=z,
            backbone_dim=backbone_dim,
            n_charts=n_charts,
            use_batch_norm=use_batch_norm,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x
            shape (batch_size, n)

        Returns
        -------
        tensor of shape (batch_size, n)
        """
        q, coords = self.encoder(x)
        x = self.decoder(q, coords)
        return x
