from typing import Tuple

from torch.utils.data import Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)


class SimCLRDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        img_size=(32, 32),
        color_dist_strength=0.5,
        crop_scale: Tuple[float, float] = (0.08, 1.0),
        horizontal_flip_prob: float = 0.0,
        grayscale_prob: float = 0.2,
    ):
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
        rnd_gray = RandomGrayscale(p=grayscale_prob)
        rnd_horizontal_flip = RandomHorizontalFlip(horizontal_flip_prob)

        transforms = [
            RandomResizedCrop(img_size, scale=crop_scale),
            rnd_color_jitter,
        ]

        if grayscale_prob > 0.0:
            transforms.append(rnd_gray)
        if horizontal_flip_prob > 0.0:
            transforms.append(rnd_horizontal_flip)

        transforms.append(ToTensor())
        self.transform = Compose(transforms)

    def __getitem__(self, index):
        return (
            self.transform(self.base_dataset[index][0]),
            self.transform(self.base_dataset[index][0]),
        )

    def __len__(self):
        return len(self.base_dataset)
