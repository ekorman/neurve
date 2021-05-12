import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    Resize,
    Normalize,
    RandomResizedCrop,
    RandomHorizontalFlip,
)

from neurve.samplers import BalancedClassBatchSampler


def get_data_loaders(
    data_root,
    resize_shape,
    val_or_test,
    num_workers,
    batch_size=None,
    n_classes=None,
    n_per_class=None,
):
    train_transform = Compose(
        [
            RandomResizedCrop(resize_shape),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = Compose(
        [
            Resize((resize_shape, resize_shape)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dset = ImageFolder(root=data_root, transform=train_transform)
    val_dset = ImageFolder(root=data_root, transform=val_transform)
    targets = train_dset.targets

    num_classes = len(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    )

    if val_or_test == "val":
        # take first half for train and val, second half for holdout test
        train_indices = [
            i
            for i in range(len(train_dset))
            if train_dset.samples[i][1] < 0.8 * num_classes / 2
        ]
        val_indices = [
            i
            for i in range(len(train_dset))
            if train_dset.samples[i][1] < num_classes / 2
            and i not in train_indices
        ]
    else:
        # split classes in half for train and val
        train_indices = [
            i
            for i in range(len(train_dset))
            if train_dset.samples[i][1] < num_classes / 2
        ]
        val_indices = [i for i in range(len(train_dset)) if i not in train_indices]

    train_dset = Subset(train_dset, train_indices)
    train_dset.targets = [targets[i] for i in train_indices]
    val_dset = Subset(val_dset, val_indices)
    val_dset.targets = [targets[i] for i in val_indices]

    if data_root == "data/CUB_200_2011/images/":
        assert (
            len(np.unique(train_dset.targets)) + len(np.unique(val_dset.targets)) == 100
            if val_or_test == "val"
            else 200
        )
    else:
        assert (
            len(np.unique(train_dset.targets)) + len(np.unique(val_dset.targets)) == 98
            if val_or_test == "val"
            else 196
        )

    if n_classes is not None:
        assert n_per_class is not None
        assert batch_size is None
        batch_sampler = BalancedClassBatchSampler(train_dset, n_classes, n_per_class)
        batch_size = n_per_class * batch_size
        train_data_loader = DataLoader(
            train_dset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        assert n_per_class is None
        assert batch_size is not None
        train_data_loader = DataLoader(
            train_dset,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=True,
        )

    val_data_loader = DataLoader(
        val_dset, batch_size=batch_size, num_workers=num_workers
    )

    return train_data_loader, val_data_loader