from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .module.base import Module


class Trainer(nn.Module):
    def __init__(self, module: Module, accumulation=1):
        super().__init__()
        self.module = module
        self.accumulation = accumulation
        self.history = []

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

    def fit(
        self,
        optimizer,
        epochs: int,
        data: DataLoader,
        val_data: DataLoader = None,
    ):
        with self.module.on_fit():
            for epoch in range(1, epochs + 1):
                self.train_one_epoch(data, epoch, optimizer)
                train_metrics_report = self.module.get_report()
                report = {"train": train_metrics_report, "epoch": epoch}
                if val_data is not None:
                    self.evaluate(val_data, epoch)
                    eval_metrics_report = self.module.get_report()
                    report["eval"] = eval_metrics_report
                self.history.append(report)

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
