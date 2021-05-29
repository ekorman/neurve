from neurve.core import Trainer
from neurve.mmd import MMDManifoldLoss, q_loss
from neurve.tsne.stats import get_cond_dist_given_perp, joint_q, kl_div


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
        E = self.net(data)
        loss = self.loss(data, E)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"train/loss": loss}


class MfldTSNETrainer(Trainer):
    def __init__(self, *args, perplexity, reg_loss_weight, q_loss_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.tsne_loss = TSNELoss(perplexity)
        self.reg_loss = MMDManifoldLoss(kernel="imq", sigma=2 / 6, device=self.device)
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight

    def _train_step(self, data):
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
