import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import wandb

from neurve.core import Trainer
from neurve.mmd import MMDManifoldLoss, q_loss
from neurve.tsne.dataset import get_datasets
from neurve.tsne.net import MfldMLP, MLP
from neurve.tsne.stats import (
    get_cond_dist_given_perp,
    joint_q,
    kl_div,
    MaxItersException,
)

sns.set(rc={"figure.figsize": (16, 8)})


def run_from_config(config=None):
    with wandb.init(config=config, project="tsne"):
        config = wandb.config

        train_dset, val_dset = get_datasets(config)
        input_dim = train_dset[0].shape[0]
        train_dl = DataLoader(train_dset, batch_size=config["batch_size"], shuffle=True)
        val_dl = DataLoader(val_dset, batch_size=config["batch_size"])

        if config.get("n_charts") is None:
            net = MLP(input_dim, config["out_dim"])
            opt = SGD(params=net.parameters(), lr=config["lr"])
            trainer = TSNETrainer(
                perplexity=config["perplexity"],
                max_var2=config["max_var2"],
                var2_tol=config["var2_tol"],
                data_loader=train_dl,
                eval_data_loader=val_dl,
                net=net,
                opt=opt,
                use_wandb=True,
                out_path=wandb.run.dir,
            )
        else:
            net = MfldMLP(input_dim, config["n_charts"])
            opt = SGD(params=net.parameters(), lr=config["lr"])
            trainer = MfldTSNETrainer(
                perplexity=config["perplexity"],
                max_var2=config["max_var2"],
                var2_tol=config["var2_tol"],
                data_loader=train_dl,
                eval_data_loader=val_dl,
                net=net,
                opt=opt,
                reg_loss_weight=config["reg_loss_weight"],
                q_loss_weight=config["q_loss_weight"],
                use_wandb=True,
                out_path=wandb.run.dir,
            )

        trainer.train(
            config["n_epochs"],
            save_ckpt_freq=config.get("save_ckpt_freq") or np.infty,
            eval_freq=config.get("eval_freq") or np.infty,
        )


class TSNELoss:
    def __init__(self, perplexity, max_var2=10000, var2_tol=1e-3):
        self.perplexity = perplexity
        self.max_var2 = max_var2
        self.var2_tol = var2_tol

    def __call__(self, X, E):
        try:
            cond_dist = get_cond_dist_given_perp(
                0, self.max_var2, self.perplexity, self.var2_tol, X
            )
        except MaxItersException:
            return None
        P = (cond_dist + cond_dist.T) / (2 * X.shape[0])
        Q = joint_q(E)

        return kl_div(P, Q)


class TSNETrainer(Trainer):
    def __init__(self, *args, perplexity, max_var2, var2_tol, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = TSNELoss(perplexity, max_var2, var2_tol)

    def _train_step(self, data):
        data = data.to(self.device)
        E = self.net(data)
        loss = self.loss(data, E)
        if loss is None:
            return None

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"train/loss": loss}

    def eval(self):
        all_embs, all_labels = np.empty((0, self.net.out_dim)), []
        for x, y in self.eval_data_loader:
            all_embs = np.concatenate(
                [all_embs, self.net(x.to(self.device)).detach().cpu().numpy()]
            )
            all_labels.extend(y)

        if self.net.out_dim == 2:
            sns.scatterplot(x=all_embs[:, 0], y=all_embs[:, 1], hue=all_labels, s=2)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            p = ax.scatter(
                all_embs[:, 0], all_embs[:, 1], all_embs[:, 2], c=all_labels, s=2
            )
            fig.colorbar(p)
        wandb.log({"embedding": wandb.Image(plt)})
        plt.clf()


class MfldTSNETrainer(Trainer):
    def __init__(
        self,
        *args,
        perplexity,
        max_var2,
        var2_tol,
        reg_loss_weight,
        q_loss_weight,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tsne_loss = TSNELoss(perplexity, max_var2, var2_tol)
        self.reg_loss = MMDManifoldLoss(kernel="imq", sigma=2 / 6, device=self.device)
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight

    def _train_step(self, data):
        data = data.to(self.device)
        q, coords, E = self.net(data)

        tsne_loss = self.tsne_loss(data, E)
        if tsne_loss is None:
            return None
        reg_loss = self.reg_loss(q, coords)
        loss = tsne_loss + self.reg_loss_weight * reg_loss
        if self.q_loss_weight != 0:
            loss += -self.q_loss_weight * q_loss(q)

        qmax = q.max(1)[0].mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "train/loss": loss.item(),
            "train/tsne_loss": tsne_loss.item(),
            "train/reg_loss": reg_loss.item(),
            "train/q_max": qmax,
        }

    def eval(self):
        all_embs, all_coords, all_q, all_labels = (
            np.empty((0, 3)),
            np.empty((0, self.n_charts, 2)),
            np.empty((0, self.n_charts)),
            [],
        )
        for x, y in self.eval_data_loader:
            q, coords, emb = self.net(x.to(self.device))
            coords = torch.sigmoid(coords)
            all_embs = np.concatenate([all_embs, emb.detach().cpu().numpy()])
            all_coords = np.concatenate([all_coords, coords.detach().cpu().numpy()])
            all_q = np.concatenate([all_q, q.detach().cpu().numpy()])
            all_labels.extend(y)

        all_labels = np.array(all_labels)
        for c in range(self.net.n_charts):
            coords = all_coords[all_q.argmax(1) == c]
            sns.scatterplot(
                x=coords[:, c, 0],
                y=coords[:, c, 1],
                hue=all_labels[all_q.argmax(1) == c],
                s=2,
            )
            wandb.log({f"chart_{c}": wandb.Image(plt)})
            plt.clf()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # p = ax.scatter(
        #     all_embs[:, 0], all_embs[:, 1], all_embs[:, 2], c=all_labels, s=2
        # )
        # fig.colorbar(p)
        # wandb.log({"embedding": wandb.Image(plt)})
        # plt.clf()
