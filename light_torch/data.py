from abc import ABC, abstractmethod
from torch import nn
from torch.utils.data.dataloader import DataLoader


class Dataset(ABC, nn.Module):

    @abstractmethod
    def get_train_data(self) -> DataLoader:
        pass

    def get_val_data(self) -> DataLoader:
        pass

    def get_test_data(self) -> DataLoader:
        pass


class ClassificationDataset(Dataset):

    @abstractmethod
    def label_to_num(self, label: str) -> int:
        pass

    @abstractmethod
    def num_to_label(self, label: int) -> str:
        pass

    @abstractmethod
    def get_label_num(self):
        pass
