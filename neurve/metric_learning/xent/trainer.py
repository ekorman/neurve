import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from neurve.distance import pdist, pdist_mfld
from neurve.metric_learning.trainer import BaseTripletTrainer
from neurve.mmd import MMDManifoldLoss, MMDLoss, q_loss


class SmoothCrossEntropy(nn.Module):
    """From https://github.com/jeromerony/dml_cross_entropy/blob/master/utils/utils.py
    """

    def __init__(self, epsilon: float = 0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        target_probs = torch.full_like(logits, self.epsilon / (logits.shape[1] - 1))
        target_probs.scatter_(1, labels.unsqueeze(1), 1 - self.epsilon)
        return F.kl_div(
            torch.log_softmax(logits, 1), target_probs, reduction="none"
        ).sum(1)


class CrossEntropyTrainer(BaseTripletTrainer):
    def __init__(
        self,
        net,
        opt,
        scheduler,
        out_path,
        data_loader,
        eval_data_loader,
        label_smoothing,
        reg_loss_weight,
        q_loss_weight,
        c,
        kernel,
        use_wandb=False,
        device=None,
    ):
        super().__init__(
            net=net,
            opt=opt,
            scheduler=scheduler,
            out_path=out_path,
            data_loader=data_loader,
            eval_data_loader=eval_data_loader,
            use_wandb=use_wandb,
            device=device,
        )
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight

        if label_smoothing is not None:
            self.xent_loss = SmoothCrossEntropy(label_smoothing)
        else:
            print("Using regular cross entropy loss")
            self.xent_loss = nn.CrossEntropyLoss()
        if self.net.is_atlas:
            self.mmd = MMDManifoldLoss(kernel="imq", sigma=net.dim_z / 6)
        else:
            self.mmd = MMDLoss(kernel, net.num_features * c / 6)

    def pdist(self, X):
        return pdist(X, X)

    def _train_step(self, data):
        x, labels = data
        labels = labels.to(self.device)
        x = x.to(self.device)

        if self.net.is_atlas:
            logits, q, coords = self.net(x)
            reg_loss = self.mmd(q, coords)
        else:
            logits, features = self.net(x)
            reg_loss = self.mmd(features)

        xent_loss = self.xent_loss(logits, labels).mean()

        loss = xent_loss + self.reg_loss_weight * reg_loss

        if self.net.is_atlas:
            loss += -self.q_loss_weight * q_loss(q)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100)
        self.opt.step()

        acc = (logits.detach().argmax(1) == labels).float().mean()

        ret_dict = {
            "train/loss": loss.item(),
            "train/xent_loss": xent_loss.item(),
            "train/reg_loss": reg_loss.item(),
            "train/acc": acc,
        }

        if self.net.is_atlas:
            ret_dict["train/q_max"] = q.detach().cpu().max(1)[0].mean()

        return ret_dict

    def _get_dists_and_targets(self):
        if self.net.is_atlas:
            return self._get_dists_and_targets_euc()
        return self._get_dists_and_targets_mfld()

    def _get_dists_and_targets_mfld(self):
        self.net.eval()
        all_q, all_coords, all_y = None, None, None
        with torch.no_grad():
            for x, y in tqdm(self.eval_data_loader):
                if all_q is None:
                    _, all_q, all_coords = self.net(x.to(self.device))
                    all_y = y
                else:
                    _, q, coords = self.net(x.to(self.device))
                    all_q = torch.cat([all_q, q])
                    all_coords = torch.cat([all_coords, coords])
                    all_y = torch.cat([all_y, y])

        all_coords = all_coords.cpu()
        all_q = all_q.cpu()
        all_q = F.one_hot(all_q.argmax(1), q.shape[1])

        dists = pdist_mfld(
            all_q.T, all_coords.transpose(0, 1), all_q.T, all_coords.transpose(0, 1)
        ).numpy()
        all_y = all_y.numpy()

        return dists, all_y

    def _get_dists_and_targets_euc(self):
        self.net.eval()
        all_emb, all_y = None, None
        tqdm.write("Getting embeddings for evaluation data")
        with torch.no_grad():
            for x, y in tqdm(self.eval_data_loader):
                if all_emb is None:
                    all_emb = self.net(x.to(self.device))[1].cpu()
                    all_y = y
                else:
                    all_emb = torch.cat([all_emb, self.net(x.to(self.device))[1].cpu()])
                    all_y = torch.cat([all_y, y])

        dists = self.pdist(all_emb).numpy()
        all_y = all_y.numpy()
        return dists, all_y
