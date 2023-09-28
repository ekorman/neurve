import os
import shutil
from pathlib import Path

from scipy.io import loadmat
from tqdm import tqdm


def main():
    base_path = Path("data/cars")
    os.makedirs(base_path)
    mat = loadmat("data/cars_annos.mat")

    annotations = list(mat["annotations"][0, :])
    class_names = list(mat["class_names"][0, :])

    for ann in tqdm(annotations):
        class_ = class_names[ann[5].item() - 1].item().replace(" ", "_")
        os.makedirs(base_path / class_, exist_ok=True)
        shutil.move(os.path.join("data", ann[0].item()), base_path / class_)


if __name__ == "__main__":
    main()
