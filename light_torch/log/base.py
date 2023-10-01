from abc import ABC, abstractmethod
from torch import nn


class BatchLogBase(nn.Module, ABC):

    @abstractmethod
    def log_at_batch(self, what, batch_num):
        pass


class EpochLogBase(nn.Module, ABC):

    @abstractmethod
    def log_at_epoch(self, what, epoch_num, stage):
        pass
