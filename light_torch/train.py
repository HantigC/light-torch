import os
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .module.base import Module
from src.utils.init import default_if_none


class Trainer(nn.Module):
    def __init__(
        self,
        module: Module,
        epochs=1,
        accumulation=1,
        optimizer=None,
        save_dir="./",
        name="trainer",
    ):
        super().__init__()

        self.optimizer = optimizer
        self.module = module
        self.accumulation = accumulation
        self.history = []
        self.save_dir = save_dir
        self.epochs = epochs
        self.start_at_epoch = 1
        self.name = name

    def train_one_epoch(self, data: DataLoader, epoch, optimizer=None):
        with self.module.on_train():
            with tqdm(total=len(data), desc=f"Train epoch {epoch: 04d}") as tbar:
                for batch_idx, batch in enumerate(data):
                    loss = self.module.train_step(
                        batch, batch_idx=batch_idx, epoch_idx=epoch
                    )
                    loss /= self.accumulation
                    loss.backward()
                    if (batch_idx % self.accumulation == 0) or (batch_idx + 1) == len(
                        data
                    ):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    logs = self.module.poplog()
                    tbar.set_postfix(logs)
                    tbar.update()

    def save_checkpoint(self, epoch):
        checkpoint = {
            "start_at_epoch": epoch,
            "epochs": self.epochs,
            "optimizer": self.optimizer.state_dict(),
            "module": self.module.state_dict(),
            "save_dir": self.save_dir,
        }
        torch.save(checkpoint, self._make_pathname())

    def _make_pathname(self):
        return os.path.join(self.save_dir, f"{self.name}_last.zip")

    def restore_checkpoint(self, save_dir, name=None):
        self.save_dir = save_dir
        name = default_if_none(name, self.name)

        checkpoint = torch.load(self._make_pathname())
        self.start_at_epoch = checkpoint["start_at_epoch"]
        self.epochs = checkpoint["epochs"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.module.load_state_dict(checkpoint["module"])
        self.save_dir = checkpoint["save_dir"]

    def fit(
        self,
        data: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        epochs: int = None,
        val_data: DataLoader = None,
    ):
        optimizer = default_if_none(optimizer, self.optimizer)
        epochs = default_if_none(epochs, self.epochs)
        with self.module.on_fit():
            for epoch in range(self.start_at_epoch, epochs + 1):
                self.train_one_epoch(data, epoch, optimizer)
                train_metrics_report = self.module.get_report()
                report = {"train": train_metrics_report, "epoch": epoch}
                if val_data is not None:
                    self.evaluate(val_data, epoch)
                    eval_metrics_report = self.module.get_report()
                    report["eval"] = eval_metrics_report
                self.history.append(report)
                self.save_checkpoint(epoch + 1)

    def evaluate(self, data: DataLoader, epoch: int = None) -> None:
        with self.module.on_val():
            with tqdm(total=len(data), desc=f"Eval epoch {epoch: 04d}") as tbar:
                for batch_idx, batch in enumerate(data):
                    self.module.val_step(batch, batch_idx=batch_idx, epoch_idx=epoch)
                    logs = self.module.poplog()
                    tbar.set_postfix(logs)
                    tbar.update()

    def get_state_dict(self):
        return {"module": self.module.get_state_dict()}
