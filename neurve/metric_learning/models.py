import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from neurve.core.models import load_from_folder


class TorchvisionEmbed(nn.Module):
    def __init__(self, backbone, embedding_dim, *args, **kwargs):
        super().__init__()
        self.backbone = getattr(models, backbone)(*args, **kwargs)
        self.last_layer = nn.Linear(1000, embedding_dim)

    def forward(self, x):
        return F.normalize(self.last_layer(self.backbone(x)))


class TorchvisionMfldEmbed(nn.Module):
    def __init__(self, backbone, dim_z, n_charts, emb_dim=None, *args, **kwargs):
        """
        Parameters
        ----------
        backbone : str
        dim_z : int
        n_charts : int
        emb_dim : int
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone = getattr(models, backbone)(*args, **kwargs)
        self.coords = nn.ModuleList([nn.Linear(1000, dim_z) for _ in range(n_charts)])
        self.q = nn.Sequential(nn.Linear(1000, n_charts), nn.Softmax(1))

        if emb_dim is not None:
            # self.embedding_heads = nn.ModuleList(
            #     [
            #         nn.Sequential(
            #             nn.Linear(dim_z, emb_dim, bias=True),
            #             nn.ReLU(),
            #             nn.Linear(emb_dim, emb_dim, bias=True),
            #         )
            #         for _ in range(n_charts)
            #     ]
            # )
            self.embedding_heads = nn.ModuleList(
                [nn.Linear(dim_z, emb_dim, bias=True) for _ in range(n_charts)]
            )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor
            input tensor of shape [N, d] where d is thee dimension of the
            ambient space.

        Returns
        -------
        q, coords : torch.tensor, torch.tensor
            q has shape [N, n_charts]
            coords has shape [N, n_charts, dim_z]
        """
        x = self.backbone(x)
        coords = torch.stack([c(x) for c in self.coords], 1)
        q = self.q(x)

        if self.emb_dim is not None:
            embs = torch.stack(
                [p(coords.select(1, i)) for i, p in enumerate(self.embedding_heads)], 1
            )
            embedding = (q.unsqueeze(2) * embs).sum(1)
            return q, coords, embedding

        return q, coords

    @classmethod
    def load_from_folder(cls, path):
        return load_from_folder(cls, path)
