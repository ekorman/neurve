# neurve

This is the repository to accompany the paper [_Self-supervised representation learning on manifolds_](https://openreview.net/forum?id=EofGDIGAhvR), to be presented at the _ICLR 2021 Workshop on Geometrical and Topological Representation Learning_.

Additionally, we implement a manifold version of triplet training, which will be expounded on in an upcoming preprint.

## Notebooks

[MSimCLR Inference](https://github.com/ekorman/neurve/blob/master/notebooks/msimclr-inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekorman/neurve/blob/master/notebooks/msimclr-inference.ipynb)

This notebook will run inference using a pre-trained Manifold SimCLR model (trained on either CIFAR10, FashionMNIST, or MNIST).

## Installation

Install via

```shell
pip install neurve
```

or, to install with [Weights & Biases](https://wandb.ai/) support, run:

```shell
pip install "neurve[wandb]"
```

You can also install from source by cloning this repository and then running, from the repo root, the command

```shell
pip install . # or pip install .[wandb]
```

The dependencies are

```
numpy>=1.17.4
torch>=1.3.1
torchvision>=0.4.2
scipy>=1.5.3 (for parsing the cars dataset annotations)
tqdm
tensorboardX
```

### Datasets

To get the datasets for metric learning (the datasets we use for representation learning are included in `torchvision.datasets`):

- CUB dataset: Download the file `CUB_200_2011.tgz` from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and decompress in the `data` folder. The folder structure should be `data/CUB_200_2011/images/`.
- cars196 dataset: run `make data/cars`.

## Training commands

### Tracking with Weights & Biases

To use [Weights & Biases](https://wandb.ai/) to log training/validation metrics and for storing model checkpoints, set the environment variable `NEURVE_TRACKER` to `wandb`. Otherwise [tensorboardX](https://github.com/lanpa/tensorboardX) will be used for metric logging and model checkpoints will be saved locally.

### Manifold SimCLR

For self-supervised training, run the command

```bash
python experiments/simclr.py \
              --dataset $DATASET \
              --backbone $BACKBONE \
              --dim_z $DIM_Z \
              --n_charts $N_CHARTS \
              --n_epochs $N_EPOCHS \
              --tau $TAU \
              --out_path $OUT_PATH # if not using Weights & Biases for tracking
```

where

- `$DATASET` is one of `"cifar"`, `"mnist"`, `"fashion_mnist"`.
- `$BACKBONE` is the name of the backbone network (in the paper we used `"resnet50"` for CIFAR10 and `"resnet18"` for MNIST and FashionMNIST).
- `$DIM_Z` and `$N_CHARTS` are the dimension and number of charts, respectively, for the manifold.
- `$N_EPOCHS` is the number of epochs to train for (in the paper we used 1,000 for CIFAR10 and 100 for MNIST and FashionMNIST).
- `$TAU` is the temperature parameter for the contrastive loss function (in the paper we used 0.5 for CIFAR10 and 1.0 for MNIST and FashionMNIST).
- `$OUT_PATH` is the path to save model checkpoints and tensorboard output.

### Manifold metric learning

To train metric learning, run the command

```bash
python experiments/triplet.py \
              --data_root $DATA_ROOT \
              --dim_z $DIM_Z \
              --n_charts $N_CHARTS \
              --out_path $OUT_PATH # if not using Weights & Biases for tracking
```

where

- `$DATA_ROOT` is the path to the data (e.g. `data/CUB_200_2011/images/` or `data/cars/`), which should be a folder of subfolders, where each subfolder has the images for one class.
- `$DIM_Z` and `$N_CHARTS` are the dimension and number of charts, respectively, for the manifold.
- `$OUT_PATH` is the path to save model checkpoints and tensorboard output.

## Citation

```
@inproceedings{
  korman2021selfsupervised,
  title={Self-supervised representation learning on manifolds},
  author={Eric O Korman},
  booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
  year={2021},
  url={https://openreview.net/forum?id=EofGDIGAhvR}
}
```
