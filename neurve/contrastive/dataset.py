from torch.utils.data import Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomApply,
    RandomGrayscale,
    RandomResizedCrop,
    ToTensor,
)


class SimCLRDataset(Dataset):
    def __init__(
        self, base_dataset, img_size=(32, 32), color_dist_strength=0.5
    ):
        """
        Parameters
        ----------
        base_dataset : Dataset
        img_size : tuple
        color_dist_strength : float
        """
        self.base_dataset = base_dataset
        # get color distortion random transform, Color distortion (see Pytorch
        # pseudocode in arXiv:2002.05709)
        color_jitter = ColorJitter(
            0.8 * color_dist_strength,
            0.8 * color_dist_strength,
            0.8 * color_dist_strength,
            0.2 * color_dist_strength,
        )
        rnd_color_jitter = RandomApply([color_jitter], p=0.8)
        rnd_gray = RandomGrayscale(p=0.2)
        self.transform = Compose(
            [
                RandomResizedCrop(img_size),
                rnd_color_jitter,
                rnd_gray,
                ToTensor(),
            ]
        )

    def __getitem__(self, index):
        return (
            self.transform(self.base_dataset[index][0]),
            self.transform(self.base_dataset[index][0]),
        )

    def __len__(self):
        return len(self.base_dataset)
