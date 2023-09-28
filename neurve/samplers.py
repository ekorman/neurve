import math
import random

import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedClassBatchSampler(Sampler):
    def __init__(self, data_source, n_classes, n_per_class):
        """
        Parameters
        ----------
        data_source : torch.utils.data.Dataset
            must have an attribute "targets", where data_source.targets[i]
            is the class label at index i.
        n_classes : int
            number of distinct classes making up a batch.
        n_per_class : int
            number of samples per class in a batch.
        """
        self.n_classes = n_classes
        self.n_per_class = n_per_class
        self.data_source = data_source
        self.labels = np.unique(data_source.targets).tolist()
        self.labels_to_idxs = {
            int(label): np.where(np.array(data_source.targets) == label)[
                0
            ].tolist()
            for label in self.labels
        }

    def __iter__(self):
        count = 0
        self.current_idxs = {label: 0 for label in self.labels}
        for idxs in self.labels_to_idxs.values():
            random.shuffle(idxs)
        while count < len(self.data_source) / (
            self.n_classes * self.n_per_class
        ):
            count += 1
            classes = random.sample(self.labels, self.n_classes)
            indices = []
            for cls in classes:
                start_idx = self.current_idxs[cls]
                end_idx = start_idx + self.n_per_class
                cls_idxs = self.labels_to_idxs[cls]
                if end_idx >= len(cls_idxs):
                    cls_idxs[cls].extend(
                        cls_idxs[: len(cls_idxs) - end_idx + 1]
                    )
                indices.extend(cls_idxs[start_idx:end_idx])
            yield indices

    def __len__(self):
        return math.ceil(
            len(self.data_source) / (self.n_classes * self.n_per_class)
        )
