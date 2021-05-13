import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from neurve.distance import pdist
from neurve.metric_learning.trainer import BaseTripletTrainer
from neurve.mmd import MMDLoss


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

        self.xent_loss = SmoothCrossEntropy(label_smoothing)
        self.mmd = MMDLoss(kernel, net.num_features * c / 6)

    def pdist(self, X):
        return pdist(X, X)

    def _train_step(self, data):
        x, labels = data
        labels = labels.to(self.device)
        x = x.to(self.device)

        logits, features = self.net(x)

        xent_loss = self.xent_loss(logits, labels).mean()
        reg_loss = self.mmd(features)
        loss = xent_loss + self.reg_loss_weight * reg_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100)
        self.opt.step()

        acc = (logits.detach().argmax(1) == labels).float().mean()

        return {
            "train/loss": loss.item(),
            "train/xent_loss": xent_loss.item(),
            "train/reg_loss": reg_loss.item(),
            "train/acc": acc,
        }

    def _get_dists_and_targets(self):
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
