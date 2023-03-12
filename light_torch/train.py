from typing import Union
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .data import Dataset
from .module.base import Module


class Trainer(nn.Module):
    def __init__(self, module: Module):
        self.module = module

    def train_one_epoch(self, dataset: Dataset, epoch, optimizer=None):
        dataloader = dataset.get_train_data()
        with self.module.on_train():
            with tqdm(total=len(dataloader)) as tbar:
                for batch in dataloader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.model(batch)
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
        with self.module.on_fit():
            for epoch in epochs:
                self.train_one_epoch(data, epoch, optimizer)

    def evaluate(self, data: Union[DataLoader, Dataset], epoch: int) -> None:
        with self.module.on_val():
            pass

    def save(self):
        pass
