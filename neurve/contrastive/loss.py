import torch
import torch.nn as nn
from neurve.distance import psim

LARGE = 1e9


class SimCLRLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        Parameters
        ----------
        z1 : torch.tensor
            tensor of shape [N, d]
        z2 : torch.tensor
            tensor of shape [N, d]

        z1[i] and z2[i] should be the embeddings of the same image under
        two augmentations.
        """
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2])
        sim = psim(z, z) / self.tau

        # set logit for a label with itself to a large negative number
        sim = sim - LARGE * torch.eye(sim.shape[0], device=z1.device)
        labels = torch.tensor(
            list(range(batch_size, 2 * batch_size)) + list(range(batch_size)),
            device=z1.device,
        )

        return self.cross_entropy(sim, labels)
