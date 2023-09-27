import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def summary_string(losses_dict, epoch, n_epochs):
    """Summary string for progress bar

    Parameters
    ----------
    losses_dict : dict
    epoch : int
        current epoch
    n_epochs : int
        total number of train epochs

    Returns
    -------
    str
    """
    ret = f"[{epoch}/{n_epochs}] " if n_epochs is not None else ""
    for k, v in losses_dict.items():
        ret += f"{k}: {v:.4f} "

    return ret


class Trainer(object):
    """
    Trainer class for training a torch.nn.Module. Takes care of boiler plate
    such as saving checkpoints and logging to tensorboard.
    """

    def __init__(
        self,
        net,
        opt,
        out_path,
        data_loader,
        net_name="net",
        eval_data_loader=None,
        device=None,
        use_wandb=False,
    ):
        """
        Parameters
        ----------
        net : torch.nn.Module
        opt : torch.optim.Optimizer
        out_path : str
            path to store various model weights and tensorboard logs
        data_loader : torch.utils.data.DataLoader
        device : torch.device
            device to use. If None then use cuda if available and cpu
            otherwise
        use_wandb : bool
        """
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.data_loader = data_loader
        self.eval_data_loader = eval_data_loader
        self.out_path = out_path
        self.net = net.to(self.device)
        self.opt = opt
        self.net_name = net_name
        self.global_steps = 0
        self.use_wandb = use_wandb

        if not use_wandb:
            self.writer = SummaryWriter(self.out_path)
        os.makedirs(out_path, exist_ok=True)

    def log_dict(self, d):
        if not self.use_wandb:
            for k, v in d.items():
                self.writer.add_scalar(k, v, self.global_steps)
        else:
            wandb.log(d)

    def train(
        self,
        n_epochs=None,
        save_ckpt_freq=np.infty,
        eval_freq=np.infty,
        n_global_steps=None,
    ):
        """Method for training. Saves tensorboard output to self.out_path and
        net and optimizer checkpoints to self.out_path.


        Parameters
        ----------
        n_epochs : int
            number of epoch to train for
        save_ckpt_freq : int
            frequency of epochs by which to save checkpoints
        eval_freq : int
            frequency of epochs by which to evaluate
        """
        self.net.train()

        for epoch in range(1, n_epochs + 1 if n_epochs is not None else 2):
            train_bar = tqdm(self.data_loader)
            for data in train_bar:
                if self.global_steps == n_global_steps:
                    return

                ret_dict = self._train_step(data)
                self.global_steps += 1

                # record loss values and update progress bar
                if ret_dict is None:
                    continue
                self.log_dict(ret_dict)
                train_bar.set_description_str(
                    summary_string(ret_dict, epoch, n_epochs)
                )

            if epoch % save_ckpt_freq == 0:
                tqdm.write(f"Saving checkpoint at step {self.global_steps}")
                self.save_ckpt(suffix=f"epoch{epoch}")

            if epoch % eval_freq == 0:
                tqdm.write("Evaluating")
                self.net.eval()
                self.eval()
                self.net.train()

    def save_ckpt(self, suffix=None):
        suffix = suffix or f"step{self.global_steps}"
        if isinstance(self.net, torch.nn.DataParallel):
            sd = self.net.module.state_dict()
        else:
            sd = self.net.state_dict()
        torch.save(
            sd,
            os.path.join(self.out_path, f"{self.net_name}_{suffix}.pth"),
        )
        torch.save(
            self.opt.state_dict(),
            os.path.join(self.out_path, f"opt_{suffix}.pth"),
        )

    def _train_step(self, data):
        """
        Parameters
        ----------
        data : list
            list of torch.Tensor objects returned by self.data_loader

        Returns
        -------
        dict
            loss dictionary
        """
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LinearTrainer(Trainer):
    def __init__(self, linear_model, opt, encoder, *args, **kwargs):
        super().__init__(
            net=linear_model, opt=opt, net_name="linear_net", *args, **kwargs
        )
        self.encoder = encoder.to(self.device)
        self.linear_model = self.net
        self.loss = torch.nn.CrossEntropyLoss()

    def _train_step(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            encoded = self.encoder.encode(x)
        if not isinstance(encoded, tuple):
            encoded = (encoded,)

        logits = self.linear_model(*encoded)
        loss = self.loss(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        preds = logits.argmax(1)
        acc = (preds == y).sum().float() / len(preds)

        return {"train/linear_loss": loss, "train/acc": acc}

    def eval(self):
        all_preds, all_labels = [], []
        for x, y in tqdm(self.eval_data_loader):
            with torch.no_grad():
                encoded = self.encoder.encode(x.to(self.device))
                if not isinstance(encoded, tuple):
                    encoded = (encoded,)
                preds = self.linear_model(*encoded).argmax(1).detach().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())
        acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(
            all_preds
        )
        print(f"Validation accuracy: {acc}")
        self.log_dict({"val/accuracy": acc})
