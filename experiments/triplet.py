import argparse
from copy import copy
import os

from torch import optim

from neurve.metric_learning.data import get_data_loaders
from neurve.metric_learning.models import TorchvisionEmbed, TorchvisionMfldEmbed
from neurve.metric_learning.trainer import (
    ManifoldTripletTrainer,
    TripletTrainer,
)


def main(
    backbone,
    dim_z,
    data_root,
    resize_shape,
    opt_name,
    lr,
    margin,
    n_charts,
    n_epochs,
    n_classes,
    n_per_class,
    save_ckpt_freq,
    eval_freq,
    reg_loss_weight,
    q_loss_weight,
    emb_dim,
    use_wandb,
    out_path,
    num_workers,
    val_or_test,
):
    assert val_or_test in ["val", "test"]

    out_path = wandb.run.dir if use_wandb else out_path

    if n_charts is not None:
        net = TorchvisionMfldEmbed(
            backbone, dim_z, n_charts, emb_dim=emb_dim, pretrained=True
        )
    else:
        # dimension is dim_z + 1 since it gets normalized to the unit sphere
        net = TorchvisionEmbed(backbone, dim_z + 1, pretrained=True)

    opt = getattr(optim, opt_name)(params=net.parameters(), lr=lr)

    train_data_loader, val_data_loader = get_data_loaders(
        data_root=data_root,
        resize_shape=resize_shape,
        val_or_test=val_or_test,
        num_workers=num_workers,
        n_classes=n_classes,
        n_per_class=n_per_class,
    )

    if n_charts is not None:
        trainer = ManifoldTripletTrainer(
            net=net,
            opt=opt,
            dim_z=dim_z,
            reg_loss_weight=reg_loss_weight,
            q_loss_weight=q_loss_weight,
            out_path=out_path,
            data_loader=train_data_loader,
            eval_data_loader=val_data_loader,
            margin=margin,
            one_hot_q=True,
            use_wandb=use_wandb,
        )
    else:
        trainer = TripletTrainer(
            net=net,
            opt=opt,
            out_path=out_path,
            data_loader=train_data_loader,
            eval_data_loader=val_data_loader,
            margin=margin,
            use_wandb=use_wandb,
        )

    trainer.train(n_epochs=n_epochs, save_ckpt_freq=save_ckpt_freq, eval_freq=eval_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="googlenet")
    parser.add_argument("--data_root", type=str, default="data/CUB_200_2011/images/")
    parser.add_argument("--dim_z", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--n_charts", type=int)
    parser.add_argument(
        "--n_classes",
        type=int,
        default=8,
        help="number of distinct classes to take when forming a batch.",
    )
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=4,
        help="number of elements per class to take when forming a batch.",
    )
    parser.add_argument("--opt_name", type=str, default="RMSprop")
    parser.add_argument("--q_loss_weight", type=float, default=0.1)
    parser.add_argument("--reg_loss_weight", type=float, default=1.0)
    parser.add_argument("--resize_shape", type=int, default=227)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--val_or_test",
        type=str,
        help="If this is equal to 'val' then a train/val split is formed from the train dataset."
        " Otherwise training is done on the entire train dataset and evaluated in the test dataset",
        default="test",
    )
    parser.add_argument("--emb_dim", type=int)

    args = parser.parse_args()

    use_wandb = os.environ.get("NEURVE_TRACKER") == "wandb"
    if use_wandb:
        import wandb

        wandb.init(project="mfld-metric")
        wandb.config.update(copy(vars(args)))
    else:
        if args.out_path is None:
            raise RuntimeError(
                "if not using wandb for logging (by setting environment variable"
                " NEURVE_TRACKER to 'wandb'), then out_path must be passed."
            )

    main(**vars(args), use_wandb=use_wandb)
