from neurve.core import Trainer
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
