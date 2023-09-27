import torch
import torch.nn as nn
from neurve.contrastive.utils import modify_resnet_model
from neurve.core.models import load_from_folder
from torchvision import models


class SimCLR(nn.Module):
    def __init__(
        self, in_channels, backbone="resnet50", dim_z=2048, proj_dim=128
    ):
        """
        Parameters
        ----------
        in_channels : int
            number of input channels
        backbone : str
            name of model in torchvision.models to use for the backbone
        dim_z : int
            encoding dimension
        proj_dim : int
            dimension of the projection head
        """
        super().__init__()
        self.encoder = modify_resnet_model(
            getattr(models, backbone)(weights=None),
            in_channels=in_channels,
            dim_z=dim_z,
        )

        self.proj_head = nn.Sequential(
            nn.Linear(dim_z, dim_z, bias=False),
            nn.ReLU(),
            nn.Linear(dim_z, proj_dim, bias=False),
        )

    def encode(self, x):
        """encodes the input tensor x"""
        return self.encoder(x)

    def forward(self, x):
        """does the composition of encoding followed by the projection head"""
        emb = self.encoder(x)
        return self.proj_head(emb)


class SimCLRMfld(nn.Module):
    def __init__(
        self, dim_z, n_charts, backbone="resnet50", proj_dim=128, in_channels=3
    ):
        """
        Parameters
        ----------
        dim_z : int
            encoding dimension
        n_charts : int
            number of charts for the atlas
        backbone : str
            name of model in torchvision.models to use for the backbone
        proj_dim : int
            dimension of the projection head
        in_channels : int
            number of input channels
        """
        super().__init__()

        self.dim_z = dim_z
        self.backbone = modify_resnet_model(
            getattr(models, backbone)(weights=None),
            in_channels=in_channels,
            dim_z=2048,
        )

        self.coords = nn.ModuleList(
            [nn.Linear(2048, dim_z) for _ in range(n_charts)]
        )
        self.q = nn.Sequential(nn.Linear(2048, n_charts), nn.Softmax(1))

        self.proj_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_z, dim_z, bias=False),
                    nn.ReLU(),
                    nn.Linear(dim_z, proj_dim, bias=False),
                )
                for _ in range(n_charts)
            ]
        )

    def encode(self, x):
        """encodes the input tensor x"""
        x = self.backbone(x)
        coords = torch.stack([c(x) for c in self.coords], 1)
        q = self.q(x)

        return q, coords

    def proj_head(self, q, coords):
        proj_comps = torch.stack(
            [p(coords.select(1, i)) for i, p in enumerate(self.proj_heads)], 1
        )
        return (q.unsqueeze(2) * proj_comps).sum(1)

    def forward(self, x):
        """does the composition of encoding followed by the projection head"""
        q, coords = self.encode(x)

        return self.proj_head(q, coords)

    @classmethod
    def load_from_folder(cls, path):
        return load_from_folder(cls, path)
