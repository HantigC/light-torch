from abc import ABC, abstractmethod
from torch import nn


class Transform(nn.Module, ABC):

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass
