from typing import Union

from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .data import Dataset
from .module.base import Module
from light_torch.utils.misc import init_if_none


class Trainer(nn.Module):
    def __init__(self, module: Module):
        self.module = module

    def train_one_epoch(self, dataset: Dataset, epoch, optimizer=None):
        dataloader = dataset.get_train_data()
        with self.module.on_train():
            with tqdm(total=len(dataloader)) as tbar:
                for batch_idx, batch in enumerate(dataloader):
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
        data: Union[DataLoader, Dataset],
        epochs: int,
        val_data: DataLoader = None,
    ):
        val_data = init_if_none(val_data, data.get_val_data())
        with self.module.on_fit():
            for epoch in epochs:
                self.train_one_epoch(data, epoch, optimizer)
                self.evaluate(val_data, epoch)

    def evaluate(self, data: Union[DataLoader, Dataset], epoch: int) -> None:
        dataloader = data.get_val_data()
        with self.module.on_val():
            for batch_idx, batch in enumerate(dataloader):
                self.module.val_step(batch, batch_idx=batch_idx, epoch_idx=epoch)

    def get_state_dict(self):
        return {"module": self.module.get_state_dict()}
