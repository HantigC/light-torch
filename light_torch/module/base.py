from abc import ABC, abstractmethod
from contextlib import contextmanager

from torch import nn


class Module(nn.Module, ABC):
    def __init__(self):
        self.trainer = None

    def predict(self, batch):
        pass

    def on_fit_begin(self):
        pass

    def on_fit_end(self):
        pass

    @contextmanager
    def on_fit(self):
        self.on_fit_begin()
        try:
            yield self
        finally:
            self.on_fit_end()

    def on_train_begin(self):
        self.train()

    @abstractmethod
    def train_step(self, batch, batch_idx, epoch_idx):
        pass

    def on_train_end(self):
        pass

    @contextmanager
    def on_train(self):
        self.on_train_begin()
        try:
            yield self
        finally:
            self.on_train_end()

    def on_val_begin(self):
        self.eval()

    def val_step(self, batch, batch_idx, epoch_idx):
        pass

    def on_val_end(self):
        pass

    @contextmanager
    def on_val(self):
        self.on_val_begin()
        try:
            yield self
        finally:
            self.on_val_end()

    def get_state_dict(self):
        pass
