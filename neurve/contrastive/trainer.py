import torch
from neurve.contrastive.loss import SimCLRLoss
from neurve.core import Trainer
from neurve.mmd import MMDManifoldLoss, q_loss


class SimCLRTrainer(Trainer):
    def __init__(self, net, opt, tau=0.5, *args, **kwargs):
        super().__init__(net=net, opt=opt, *args, **kwargs)
        self.loss = SimCLRLoss(tau=tau)

    def _train_step(self, data):
        z1, z2 = self.net(data[0].to(self.device)), self.net(
            data[1].to(self.device)
        )
        loss = self.loss(z1, z2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"train/loss": loss}


class SimCLRMfldTrainer(SimCLRTrainer):
    def __init__(
        self,
        net,
        opt,
        reg_loss_weight,
        q_loss_weight=0,
        tau=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(net, opt, tau, *args, **kwargs)
        self.reg_loss_weight = reg_loss_weight
        self.reg_loss = MMDManifoldLoss(
            kernel="imq", sigma=self.net.dim_z / 6, device=self.device
        )
        self.q_loss_weight = q_loss_weight

    def _train_step(self, data):
        q1, coords1 = self.net.encode(data[0].to(self.device))
        q2, coords2 = self.net.encode(data[1].to(self.device))

        coords1 = torch.clamp(coords1, -10, 10)
        coords2 = torch.clamp(coords2, -10, 10)

        reg_loss = 0.5 * self.reg_loss(q1, coords1) + 0.5 * self.reg_loss(
            q2, coords2
        )

        coords1 = torch.sigmoid(coords1)
        coords2 = torch.sigmoid(coords2)
        z1, z2 = self.net.proj_head(q1, coords1), self.net.proj_head(
            q2, coords2
        )
        simclr_loss = self.loss(z1, z2)

        loss = simclr_loss + self.reg_loss_weight * reg_loss

        if self.q_loss_weight != 0:
            loss += -self.q_loss_weight * (0.5 * q_loss(q1) + 0.5 * q_loss(q2))

        qmax = 0.5 * q1.max(1)[0].mean() + 0.5 * q2.max(1)[0].mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "train/loss": loss.item(),
            "train/simclr_loss": simclr_loss.item(),
            "train/reg_loss": reg_loss.item(),
            "train/q_max": qmax.item(),
        }
