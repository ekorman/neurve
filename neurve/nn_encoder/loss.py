from collections import defaultdict

import torch

from neurve.distance import pdist_mfld


def loss_at_a_point(
    point: torch.Tensor,
    neighbors: torch.Tensor,
    q_point: torch.Tensor,
    q_neighbors: torch.Tensor,
    q_non_neighbors: torch.Tensor,
    coords_point: torch.Tensor,
    coords_neighbors: torch.Tensor,
    coords_non_neighbors: torch.Tensor,
    scale: torch.Tensor,
    c: float,
):
    """
    Parameters
    ----------
    point
        shape [n]
    neighbors
        shape [n_neighbors, n]
    q_point
        shape [n_charts]
    q_neighbors
        shape [n_neighbors, n_charts]
    q_non_neighbors
        shape [n_non_neighbors, n_charts]
    coords_point
        shape [n_charts, z]
    coords_neighbors
        shape [n_neighbors, n_charts, z]
    coords_non_neighbors
        shape [n_non_neighbors, n_charts, z]
    scale
        shape [1]
    """
    # squared distance between og_point and neighbors
    og_neighbors_dist2 = ((point - neighbors) ** 2).sum(
        1
    )  # shape [n_neighbors]

    # squared distance in latent space between point and neighbors
    latent_neighbors_dist2 = pdist_mfld(
        q_point.unsqueeze(1),
        coords_point.unsqueeze(1),
        q_neighbors.T,
        coords_neighbors.transpose(0, 1),
    )  # shape [1, n_neighbors]

    latent_non_neighbors_dist2 = pdist_mfld(
        q_point.unsqueeze(1),
        coords_point.unsqueeze(1),
        q_non_neighbors.T,
        coords_non_neighbors.transpose(0, 1),
    )  # shape [1, n_non_neighbors]

    loss_neighbors = torch.abs(
        og_neighbors_dist2 - scale * latent_neighbors_dist2
    ).sum()

    # encourage non neighbors to be further than neighbors
    loss_non_neighbors = -(
        latent_non_neighbors_dist2 - latent_neighbors_dist2.max()
    )

    return {
        "loss_neighbors": loss_neighbors,
        "loss_non_neighbors": loss_non_neighbors,
        "loss": loss_neighbors + c * loss_non_neighbors,
    }


# TODO: better way to do this in a batch?
def loss(
    point: torch.Tensor,
    neighbors: torch.Tensor,
    q_point: torch.Tensor,
    q_neighbors: torch.Tensor,
    q_non_neighbors: torch.Tensor,
    coords_point: torch.Tensor,
    coords_neighbors: torch.Tensor,
    coords_non_neighbors: torch.Tensor,
    scale: torch.Tensor,
    c: float,
):
    """
    Parameters
    ----------
    point
        shape [b, n]
    neighbors
        shape [b, n_neighbors, n]
    q_point
        shape [b, n_charts]
    q_neighbors
        shape [b, n_neighbors, n_charts]
    q_non_neighbors
        shape [b, n_non_neighbors, n_charts]
    coords_point
        shape [b, n_charts, z]
    coords_neighbors
        shape [b, n_neighbors, n_charts, z]
    coords_non_neighbors
        shape [b, n_non_neighbors, n_charts, z]
    scale
        shape [1]
    """
    loss = defaultdict(lambda: 0)
    for p, n, q_p, q_n, q_nn, c_p, c_n, c_nn in zip(
        point,
        neighbors,
        q_point,
        q_neighbors,
        q_non_neighbors,
        coords_point,
        coords_neighbors,
        coords_non_neighbors,
    ):
        loss_pt = loss_at_a_point(
            p, n, q_p, q_n, q_nn, c_p, c_n, c_nn, scale, c
        )

        for k, v in loss_pt.items():
            loss[k] += v

    for k, v in loss.items():
        loss[k] /= point.shape[0]
    return loss
