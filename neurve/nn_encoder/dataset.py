import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


class NNDataset(Dataset):
    def __init__(self, data: np.ndarray, n_neighbors: int) -> None:
        """dataset that returns a point together with neighbors and non neighbors
        and non-neighbors
        """
        # for non-neighbors randomly sample from data that are not within the 2 * n_neighbors
        # closest points
        self.data = data
        nbrs = NearestNeighbors(n_neighbors=2 * n_neighbors).fit(data)
        self.n_neighbors = n_neighbors
        distances, self.indices = nbrs.kneighbors()

        # check that distances are increasing
        assert np.array_equal(distances, np.sort(distances))

    def sample_non_neighbor(self, index: int) -> int:
        ret = []
        while len(ret) < self.n_neighbors:
            idx = random.choice(range(len(self.data)))
            if idx not in self.indices[index] and idx != index:
                ret.append(idx)
        return ret

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The first array is the point. shape: self.data.shape[1])
        The second array are the nearest neighbors. shape: (self.n_neighbors, self.data.shape[1])
        The third array are non-neighbors. shape: (self.n_neighbors, self.data.shape[1])
        """
        point = self.data[index]
        neighbors = self.data[self.indices[index][: self.n_neighbors]]
        non_neighbors = self.data[self.sample_non_neighbor(index)]

        return point, neighbors, non_neighbors
