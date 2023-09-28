import torch
import torch.nn as nn
import torch.nn.functional as F
from neurve.distance import pdist, pdist_mfld


def loss_from_dist_matrix(dist_matrix, labels, margin):
    same = labels.reshape(1, -1) == labels.reshape(-1, 1)
    diff = ~same
    same = same.fill_diagonal_(False)
    anc_idxs, pos_idxs, neg_idxs = torch.where(
        same.unsqueeze(2) * diff.unsqueeze(1)
    )

    ap_minus_an = (
        dist_matrix[anc_idxs, pos_idxs] - dist_matrix[anc_idxs, neg_idxs]
    )
    losses = ap_minus_an + margin

    n_eg = (losses > 0).sum()
    return F.relu(losses).mean(), n_eg


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, z, labels):
        dist_matrix = pdist(z, z)

        return loss_from_dist_matrix(dist_matrix, labels, self.margin)


class ManifoldTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, q, coords, labels):
        """
        Parameters
        ----------
        q : torch.tensor
            shape [n, nc]
        coords : torch.tensor
            shape [n, nc, d]
        labels : torch.tensor
            shape n
        """
        dist_matrix = pdist_mfld(
            q.transpose(0, 1),
            coords.transpose(0, 1),
            q.transpose(0, 1),
            coords.transpose(0, 1),
        )

        return loss_from_dist_matrix(dist_matrix, labels, self.margin)
