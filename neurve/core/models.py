import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_from_folder(cls, path):
    """
    Parameters
    ----------
    cls : type
        should inherit from nn.Module
    path : str
        path to directory containing a file called "config.json" that provides
        the arguments to the constructor of cls and a file ending in "pth" that
        is the serialized state dict to load

    Returns
    -------
    cls
        loaded net
    """
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Could not find file {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    net_fnames = [f for f in os.listdir(path) if f.endswith("pth")]
    if len(net_fnames) > 1:
        raise RuntimeError(f"Found more than one state dict file in {path}")
    if len(net_fnames) == 0:
        raise RuntimeError(f"Did not find a state dict file in {path}")
    net_path = os.path.join(path, net_fnames[0])

    net = cls(**config)
    net.load_state_dict(
        torch.load(
            net_path,
            map_location=None
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
    )
    return net.eval()


class CoordLinear(nn.Module):
    def __init__(self, dim_z, n_charts, out_dim, one_hot_q=True):
        super().__init__()
        self.out_dim = out_dim
        self.weights = nn.parameter.Parameter(
            torch.Tensor(n_charts, out_dim, dim_z)
        )
        self.biases = nn.parameter.Parameter(torch.Tensor(n_charts, out_dim))
        self.one_hot_q = one_hot_q

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, q, coords):
        """
        Parameters
        ----------
        q : torch.tensor
            shape [n, n_charts]
        coords : torch.tensor
            shape [n, n_charts, dim_z]

        Returns
        -------
        torch.tensor
            shape [n, self.out_dim]
        """
        # shape [n, n_charts, self.out_dim]
        coords = torch.sigmoid(coords)
        if self.one_hot_q:
            q = F.one_hot(q.argmax(1), q.shape[1])
        ret = (
            self.weights @ coords.unsqueeze(-1)
            + self.biases.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        return (q.unsqueeze(-1) * ret).sum(1)
