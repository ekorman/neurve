import argparse
from copy import copy
import math
import os

from torch.optim import lr_scheduler, SGD

from neurve.metric_learning.data import get_data_loaders
from neurve.metric_learning.xent import CrossEntropyTrainer, resnet50


def main(
    backbone,
    data_root,
    resize_shape,
    lr,
    label_smoothing,
    c,
    reg_loss_weight,
    q_loss_weight,
    num_features,
    n_charts,
    dim_z,
    dropout,
    n_epochs,
    batch_size,
    save_ckpt_freq,
    eval_freq,
    use_wandb,
    out_path,
    num_workers,
    val_or_test,
    kernel,
):
    assert val_or_test in ["val", "test"]

    out_path = wandb.run.dir if use_wandb else out_path

    train_data_loader, val_data_loader = get_data_loaders(
        data_root=data_root,
        resize_shape=resize_shape,
        val_or_test=val_or_test,
        num_workers=num_workers,
        batch_size=batch_size,
        method="xent",
    )

    net = resnet50(
        pretrained=True,
        num_classes=len(set(train_data_loader.dataset.targets)),
        num_features=num_features,
        n_charts=n_charts,
        dim_z=dim_z,
        dropout=dropout,
        set_bn_eval=True,
    )

    parameters = [
        {"params": [par for par in net.parameters() if par.dim() != 1]},
        {
            "params": [par for par in net.parameters() if par.dim() == 1],
            "weight_decay": 0,
        },
    ]
    opt = SGD(parameters, lr=lr, nesterov=False, weight_decay=5e-4)
    warm_cosine = lambda i: min(
        (i + 1) / 100,
        (1 + math.cos(math.pi * i / (n_epochs * len(train_data_loader)))) / 2,
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=warm_cosine)

    trainer = CrossEntropyTrainer(
        net=net,
        opt=opt,
        scheduler=scheduler,
        out_path=out_path,
        data_loader=train_data_loader,
        eval_data_loader=val_data_loader,
        label_smoothing=label_smoothing,
        c=c,
        kernel=kernel,
        reg_loss_weight=reg_loss_weight,
        q_loss_weight=q_loss_weight,
        use_wandb=use_wandb,
    )

    trainer.train(n_epochs=n_epochs, save_ckpt_freq=save_ckpt_freq, eval_freq=eval_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="googlenet")
    parser.add_argument("--data_root", type=str, default="data/CUB_200_2011/images/")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--label_smoothing")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--reg_loss_weight", type=float, default=0.1)
    parser.add_argument("--q_loss_weight", type=float)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="imq")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--resize_shape", type=int, default=256)
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_features", type=int)
    parser.add_argument("--n_charts", type=int)
    parser.add_argument("--dim_z", type=int)

    args = parser.parse_args()

    use_wandb = os.environ.get("NEURVE_TRACKER") == "wandb"
    if use_wandb:
        import wandb

        wandb.init(project="xent-metric")
        wandb.config.update(copy(vars(args)))
    else:
        if args.out_path is None:
            raise RuntimeError(
                "if not using wandb for logging (by setting environment variable"
                " NEURVE_TRACKER to 'wandb'), then out_path must be passed."
            )

    main(**vars(args), use_wandb=use_wandb)
