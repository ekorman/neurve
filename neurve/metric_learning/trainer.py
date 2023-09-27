import numpy as np
import torch
import torch.nn.functional as F
from neurve.core import Trainer
from neurve.distance import pdist, pdist_mfld
from neurve.metric_learning.loss import ManifoldTripletLoss, TripletLoss
from neurve.metric_learning.metrics import retrieval_metrics
from neurve.mmd import MMDManifoldLoss, q_loss
from tqdm import tqdm


class BaseTripletTrainer(Trainer):
    def __init__(
        self,
        net,
        opt,
        out_path,
        data_loader,
        eval_data_loader,
        use_wandb=False,
        device=None,
    ):
        super().__init__(
            net=net,
            opt=opt,
            out_path=out_path,
            data_loader=data_loader,
            eval_data_loader=eval_data_loader,
            device=device,
            use_wandb=use_wandb,
        )

    def pdist(self, X):
        raise NotImplementedError

    def _train_step(self, data):
        raise NotImplementedError

    def log_metrics(self, dists, all_y):
        """Logs metrics

        Parameters
        ----------
        dists : np.ndarray
        all_y : np.ndarray
        """
        all_y = np.ones((dists.shape[0], 1)) @ all_y.reshape(1, -1)
        np.fill_diagonal(dists, np.infty)
        sorted_indices = np.argsort(dists)

        # sort dists and labels by distance
        dists = np.take_along_axis(dists, sorted_indices, -1)
        all_y = np.take_along_axis(all_y, sorted_indices, -1)

        mets = retrieval_metrics(all_y[:, -1], all_y[:, :-1])
        print(f"Validation metrics: {mets}")
        self.log_dict({f"val/{k}": v for k, v in mets.items()})

    def _get_dists_and_targets(self):
        raise NotImplementedError

    def eval(self):
        """Compute accuracy of nearest neighbor matching"""
        dists, all_y = self._get_dists_and_targets()
        self.log_metrics(dists, all_y)


class TripletTrainer(BaseTripletTrainer):
    def __init__(
        self,
        net,
        opt,
        out_path,
        data_loader,
        eval_data_loader,
        margin,
        use_wandb=False,
        device=None,
    ):
        super().__init__(
            net=net,
            opt=opt,
            out_path=out_path,
            data_loader=data_loader,
            eval_data_loader=eval_data_loader,
            use_wandb=use_wandb,
            device=device,
        )
        self.loss = TripletLoss(margin=margin)

    def pdist(self, X):
        return pdist(X, X)

    def _train_step(self, data):
        x, labels = data
        x = x.to(self.device)

        encodings = self.net(x)

        loss, n_egs = self.loss(encodings, labels)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"train/loss": loss.item(), "train/n_egs": n_egs}

    def _get_dists_and_targets(self):
        self.net.eval()
        all_emb, all_y = None, None
        tqdm.write("Getting embeddings for evaluation data")
        with torch.no_grad():
            for x, y in tqdm(self.eval_data_loader):
                if all_emb is None:
                    all_emb = self.net(x.to(self.device)).cpu()
                    all_y = y
                else:
                    all_emb = torch.cat(
                        [all_emb, self.net(x.to(self.device)).cpu()]
                    )
                    all_y = torch.cat([all_y, y])

        dists = self.pdist(all_emb).numpy()
        all_y = all_y.numpy()
        return dists, all_y


class ManifoldTripletTrainer(BaseTripletTrainer):
    def __init__(
        self,
        net,
        opt,
        out_path,
        data_loader,
        eval_data_loader,
        margin,
        dim_z,
        reg_loss_weight,
        q_loss_weight,
        use_wandb=False,
        device=None,
        one_hot_q=True,
    ):
        super().__init__(
            net=net,
            opt=opt,
            out_path=out_path,
            data_loader=data_loader,
            eval_data_loader=eval_data_loader,
            use_wandb=use_wandb,
            device=device,
        )

        self.triplet_loss = ManifoldTripletLoss(margin=margin)
        self.reg_loss = MMDManifoldLoss(
            kernel="imq", sigma=dim_z / 6, device=self.device
        )
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight
        self.one_hot_q = one_hot_q

    def _train_step(self, data):
        x, labels = data
        x = x.to(self.device)

        q, coords = self.net(x)

        triplet_loss, n_egs = self.triplet_loss(q, coords, labels)

        reg_loss = self.reg_loss(q, coords)

        loss = triplet_loss + self.reg_loss_weight * reg_loss

        if self.q_loss_weight != 0:
            loss += -self.q_loss_weight * q_loss(q)

        qmax = q.max(1)[0].mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "train/loss": loss.item(),
            "train/triplet_loss": triplet_loss.item(),
            "train/reg_loss": reg_loss.item(),
            "train/qa_max": qmax,
            "train/n_egs": n_egs,
        }

    def pdist(self, all_q, all_coords):
        return pdist_mfld(
            all_q.T,
            all_coords.transpose(0, 1),
            all_q.T,
            all_coords.transpose(0, 1),
        )

    def _get_dists_and_targets(self):
        self.net.eval()
        all_q, all_coords, all_y = None, None, None
        with torch.no_grad():
            for x, y in tqdm(self.eval_data_loader):
                if all_q is None:
                    all_q, all_coords = self.net(x.to(self.device))
                    all_y = y
                else:
                    q, coords = self.net(x.to(self.device))
                    all_q = torch.cat([all_q, q])
                    all_coords = torch.cat([all_coords, coords])
                    all_y = torch.cat([all_y, y])

        all_coords = all_coords.cpu()
        all_q = all_q.cpu()
        if self.one_hot_q:
            all_q = F.one_hot(all_q.argmax(1), q.shape[1])

        dists = self.pdist(all_q, all_coords).numpy()
        all_y = all_y.numpy()

        return dists, all_y
