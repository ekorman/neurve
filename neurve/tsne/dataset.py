import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_datasets(config):
    if config["dataset"] == "mnist":
        train_dset = FlatMNIST(train=True, download=True, root=config["data_root"])
        val_dset = FlatMNIST(
            train=False, download=True, root=config["data_root"], return_labels=True
        )
    elif config["dataset"] == "macosko":
        train_dset = PickleDataset(
            path="data/macosko_2015_unnormalized.pkl", return_labels=False
        )
        val_dset = PickleDataset(
            path="data/macosko_2015_unnormalized.pkl", return_labels=True
        )

    return train_dset, val_dset


class PickleDataset(Dataset):
    def __init__(self, path, return_labels=False):
        self.return_labels = return_labels
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.X = torch.Tensor(data["X"].astype(np.float32))
        self.y = data["y"]

    def __getitem__(self, index):
        if self.return_labels:
            return self.X[index], self.y[index]
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]


class FlatMNIST(MNIST):
    def __init__(self, *args, return_labels=False, **kwargs):
        super().__init__(*args, **kwargs, transform=lambda x: ToTensor()(x).flatten())
        self.return_labels = return_labels

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        return ret if self.return_labels else ret[0]
