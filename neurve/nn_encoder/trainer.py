import torch

from neurve.core import Trainer
from neurve.nn_encoder.loss import loss
from neurve.nn_encoder.models import MfldEncoder

# loss: at a point, take the nearest neighbors and look at squared error between distance in
# og space and distance in latent space (with a learned scale parameter) and then sample poitns
# that are not NN and add a term that has the distance from the point to those bigger than some margin
# from the max (or smooth maximum) with the distance to the nearest neighbors (but capped). actually
# maybe don't need to cap since vectors will be restricted to inside unit ball. so then maybe just have term
# that encourages these other points to be far away and then don't need to worry about max


class NNEncoderTrainer(Trainer):
    def __init__(
        self,
        net: MfldEncoder,
        opt: torch.optim.Optimizer,
        out_path: str,
        reg_loss_weight: float,
        c: float,
        data_loader: torch.utils.data.DataLoader,
        net_name: str = "net",
        eval_data_loader: torch.utils.data.DataLoader = None,
        device: torch.device = None,
        q_loss_weight: float = 0.0,
        use_wandb: bool = False,
    ):
        super().__init__(
            net=net,
            opt=opt,
            out_path=out_path,
            data_loader=data_loader,
            net_name=net_name,
            eval_data_loader=eval_data_loader,
            device=device,
            use_wandb=use_wandb,
        )
        self.reg_loss_weight = reg_loss_weight
        self.q_loss_weight = q_loss_weight
        self.scale = torch.rand(1, requires_grad=True, device=self.device)
        self.c = c

    def _train_step(self, data):
        batch_size = data[0].shape[0]

        points, neighbors, non_neighbors = data

        batch_size, n_neighbors, _ = neighbors.shape

        all_points = torch.cat(
            [points, neighbors.flatten(0, 1), non_neighbors.flatten(0, 1)],
            dim=0,
        )
        q, coords = self.net(all_points)

        sections = [
            batch_size,
            n_neighbors * batch_size,
            n_neighbors * batch_size,
        ]

        q_point, q_neighbors, q_non_neighbors = q.split(sections)
        q_neighbors = q_neighbors.unflatten(0, (batch_size, n_neighbors))
        q_non_neighbors = q_non_neighbors.unflatten(
            0, (batch_size, n_neighbors)
        )

        coords_point, coords_neighbors, coords_non_neighbors = coords.split(
            sections
        )
        coords_neighbors = coords_neighbors.unflatten(
            0, (batch_size, n_neighbors)
        )
        coords_non_neighbors = coords_non_neighbors.unflatten(
            0, (batch_size, n_neighbors)
        )

        loss_dict = loss(
            point=points,
            neighbors=neighbors,
            q_point=q_point,
            q_neighbors=q_neighbors,
            q_non_neighbors=q_non_neighbors,
            coords_point=coords_point,
            coords_neighbors=coords_neighbors,
            coords_non_neighbors=coords_non_neighbors,
            scale=self.scale,
            c=self.c,
        )

        self.opt.zero_grad()
        loss_dict["loss"].backward()
        self.opt.step()

        return loss_dict
