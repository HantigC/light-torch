from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .module.base import Module


class Trainer(nn.Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def train_one_epoch(self, data: DataLoader, epoch, optimizer=None):
        with self.module.on_train():
            with tqdm(total=len(data)) as tbar:
                for batch_idx, batch in enumerate(data):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.module.train_step(
                        batch, batch_idx=batch_idx, epoch_idx=epoch
                    )
                    loss.backward()
                    optimizer.step()
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
                if val_data is not None:
                    self.evaluate(val_data, epoch)

    def evaluate(self, data: DataLoader, epoch: int = None) -> None:
        with self.module.on_val():
            with tqdm(total=len(data)) as tbar:
                for batch_idx, batch in enumerate(data):
                    self.module.val_step(batch, batch_idx=batch_idx, epoch_idx=epoch)
                    tbar.update()

    def get_state_dict(self):
        return {"module": self.module.get_state_dict()}
