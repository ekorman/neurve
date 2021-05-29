import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader
import wandb

from neurve.core import Trainer
from neurve.mmd import MMDManifoldLoss, q_loss
from neurve.tsne.dataset import FlatMNIST
from neurve.tsne.net import MfldMLP, MLP
from neurve.tsne.stats import get_cond_dist_given_perp, joint_q, kl_div


def run_from_conf(conf):
    wandb.init(project="tsne")
    wandb.config.update(conf)

    train_dset = FlatMNIST(train=True, download=True, root=conf["data_root"])
    train_dl = DataLoader(train_dset, batch_size=conf["batch_size"])

    val_dset = FlatMNIST(train=False, download=True, root=conf["data_root"])
    val_dl = DataLoader(val_dset, batch_size=conf["batch_size"])
    if conf.get("n_charts") is None:
        net = MLP(28 * 28, conf["out_dim"])
        opt = SGD(params=net.parameters(), lr=conf["lr"])
        trainer = TSNETrainer(
            perplexity=conf["perplexity"],
            data_loader=train_dl,
            eval_data_loader=val_dl,
            net=net,
            opt=opt,
            use_wandb=True,
            out_path=wandb.run.dir,
        )
    else:
        net = MfldMLP(28 * 28, conf["n_charts"])
        opt = SGD(params=net.parameters(), lr=conf["lr"])
        trainer = MfldTSNETrainer(
            perplexity=conf["perplexity"],
            data_loader=train_dl,
            net=net,
            opt=opt,
            reg_loss_weight=conf["reg_loss_weight"],
            q_loss_weight=conf["q_loss_weight"],
            use_wandb=True,
            out_path=wandb.run.dir,
        )

    trainer.train(conf["n_epochs"])


class TSNELoss:
    def __init__(self, perplexity):
        self.perplexity = perplexity

    def __call__(self, X, E):
        cond_dist = get_cond_dist_given_perp(0, 100, self.perplexity, 1e-4, X)
        P = (cond_dist + cond_dist.T) / (2 * X.shape[0])
        Q = joint_q(E)

        return kl_div(P, Q)


class TSNETrainer(Trainer):
    def __init__(self, *args, perplexity, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = TSNELoss(perplexity)

    def _train_step(self, data):
        data = data.to(self.device)
        E = self.net(data)
        loss = self.loss(data, E)

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
            all_labels.extend(y.detach().cpu().numpy().tolist())

        all_labels = np.array(all_labels)
        if self.net.out_dim == 2:
            plt.scatter(all_embs[:, 0], all_embs[:, 1], c=all_labels, s=5)
            plt.colorbar()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(
                all_embs[:, 0], all_embs[:, 1], all_embs[:, 2], c=all_labels, s=2
            )
            ax.colorbar()
        wandb.log({"embedding": wandb.Image(plt)})


class MfldTSNETrainer(Trainer):
    def __init__(self, *args, perplexity, reg_loss_weight, q_loss_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.tsne_loss = TSNELoss(perplexity)
        self.reg_loss = MMDManifoldLoss(kernel="imq", sigma=2 / 6, device=self.device)
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight

    def _train_step(self, data):
        data = data.to(self.device)
        q, coords, E = self.net(data)

        tsne_loss = self.tsne_loss(data, E)
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
            np.empty((0, 4, 2)),
            np.empty((0, 4)),
            [],
        )
        for x, y in self.eval_data_loader:
            q, coords, emb = self.net(x.to(self.device))
            all_embs = np.concatenate([all_embs, emb.detach().cpu().numpy()])
            all_coords = np.concatenate([all_coords, coords.detach().cpu().numpy()])
            all_q = np.concatenate([all_q, q.detach().cpu().numpy()])
            all_labels.extend(y.detach().cpu().numpy().tolist())

        all_labels = np.array(all_labels)
        for c in range(self.n_charts):
            coords = all_coords[all_q.argmax(1) == c]
            plt.scatter(
                coords[:, c, 0],
                coords[:, c, 1],
                c=all_labels[all_q.argmax(1) == c],
                s=5,
            )
            plt.colorbar()
            wandb.log({f"chart_{c}": plt})

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(all_embs[:, 0], all_embs[:, 1], all_embs[:, 2], c=all_labels, s=2)
        ax.colorbar()
        wandb.log({"embedding": wandb.Image(plt)})
