import argparse
import logging
import os
from copy import copy

import torch
from neurve.contrastive.dataset import SimCLRDataset
from neurve.contrastive.models import SimCLR, SimCLRMfld
from neurve.contrastive.trainer import SimCLRMfldTrainer, SimCLRTrainer
from neurve.core.models import CoordLinear
from neurve.core.trainer import LinearTrainer
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor

data_root = "data/"


def main(
    backbone,
    dim_z,
    proj_dim,
    opt_name,
    lr,
    weight_decay,
    save_ckpt_freq,
    eval_freq,
    n_epochs,
    tau,
    batch_size,
    n_charts,
    reg_loss_weight,
    q_loss_weight,
    linear_lr,
    linear_opt_name,
    linear_weight_decay,
    dataset,
    use_wandb,
    out_path,
    num_workers,
    val_or_test,
):
    assert val_or_test in ["val", "test"]

    out_path = wandb.run.dir if use_wandb else out_path
    dset_class = {
        "cifar": CIFAR10,
        "mnist": MNIST,
        "fashion_mnist": FashionMNIST,
    }[dataset]

    # get the net
    if n_charts is None:
        net = SimCLR(
            backbone=backbone,
            in_channels=3 if dataset == "cifar" else 1,
            dim_z=dim_z,
            proj_dim=proj_dim,
        )
    else:
        net = SimCLRMfld(
            backbone=backbone,
            in_channels=3 if dataset == "cifar" else 1,
            dim_z=dim_z,
            n_charts=n_charts,
            proj_dim=proj_dim,
        )

    if use_wandb:
        wandb.watch(net)

    # setup the SimCLR training dataset
    dset = dset_class(download=True, train=True, root=data_root)
    train_size = int(0.8 * len(dset))
    if val_or_test == "val":
        train_dset, _ = random_split(
            dset,
            [train_size, len(dset) - train_size],
            generator=torch.Generator().manual_seed(18),
        )
    else:
        train_dset = dset

    train_dset = SimCLRDataset(train_dset)

    train_data_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    opt_kwargs = {} if weight_decay is None else {"weight_decay": weight_decay}
    opt = getattr(optim, opt_name)(
        params=net.parameters(), lr=lr, **opt_kwargs
    )

    # setup the trainer
    if n_charts is None:
        trainer = SimCLRTrainer(
            net=net,
            tau=tau,
            opt=opt,
            out_path=out_path,
            data_loader=train_data_loader,
            use_wandb=use_wandb,
        )
    else:
        trainer = SimCLRMfldTrainer(
            net=net,
            tau=tau,
            opt=opt,
            reg_loss_weight=reg_loss_weight,
            q_loss_weight=q_loss_weight,
            out_path=out_path,
            data_loader=train_data_loader,
            use_wandb=use_wandb,
        )

    trainer.train(n_epochs=n_epochs, save_ckpt_freq=save_ckpt_freq)

    # run linear evaluation
    net = trainer.net.eval()

    if n_charts is None:
        linear_model = torch.nn.Linear(dim_z, 10)
    else:
        linear_model = CoordLinear(dim_z, n_charts, 10, one_hot_q=True)

    linear_opt_kwargs = (
        {}
        if linear_weight_decay is None
        else {"weight_decay": linear_weight_decay}
    )
    linear_opt = getattr(optim, linear_opt_name)(
        params=linear_model.parameters(), lr=linear_lr, **linear_opt_kwargs
    )

    linear_dset = dset_class(
        download=True, train=True, root=data_root, transform=ToTensor()
    )
    if val_or_test == "val":
        linear_train_dset, val_dset = random_split(
            linear_dset,
            [train_size, len(linear_dset) - train_size],
            generator=torch.Generator().manual_seed(18),
        )
    else:
        linear_train_dset = linear_dset
        val_dset = dset_class(
            download=True, train=False, root=data_root, transform=ToTensor()
        )

    linear_train_data_loader = DataLoader(
        linear_train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_data_loader = DataLoader(
        val_dset, batch_size=batch_size, num_workers=num_workers
    )

    logging.info("Training linear model")

    trainer = LinearTrainer(
        linear_model,
        linear_opt,
        net,
        out_path=out_path,
        data_loader=linear_train_data_loader,
        eval_data_loader=val_data_loader,
        use_wandb=use_wandb,
    )
    trainer.train(
        n_epochs=n_epochs, save_ckpt_freq=save_ckpt_freq, eval_freq=eval_freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--dim_z", type=int)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--opt_name", type=str, default="Adam")
    parser.add_argument("--weight_decay", type=float, default=2.0e-6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--n_charts", type=int)
    parser.add_argument("--reg_loss_weight", type=float, default=20)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--linear_opt_name", type=str, default="Adam")
    parser.add_argument("--linear_weight_decay", type=float, default=0.0)
    parser.add_argument("--q_loss_weight", type=float, default=0.1)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--val_or_test",
        type=str,
        help="If this is equal to 'val' then a train/val split is formed from the train dataset."
        " Otherwise training is done on the entire train dataset and evaluated in the test dataset",
        default="test",
    )

    args = parser.parse_args()

    use_wandb = os.environ.get("NEURVE_TRACKER") == "wandb"
    if use_wandb:
        import wandb

        wandb.init(project="mfld-simclr")
        wandb.config.update(copy(vars(args)))
    else:
        if args.out_path is None:
            raise RuntimeError(
                "if not using wandb for logging (by setting environment variable"
                " NEURVE_TRACKER to 'wandb'), then out_path must be passed."
            )

    main(**vars(args), use_wandb=use_wandb)
