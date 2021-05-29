from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class FlatMNIST(MNIST):
    def __init__(self, *args, return_labels=False, **kwargs):
        super().__init__(*args, **kwargs, transform=lambda x: ToTensor()(x).flatten())
        self.return_labels = return_labels

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        return ret if self.return_labels else ret[0]
