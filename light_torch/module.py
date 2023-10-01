from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from torch import nn
from .eval.collector import WindowCollector, StepCollector


class Module(nn.Module, ABC):
    def __init__(self, model=None, device=None, window_size=40):
        super().__init__()
        self.trainer = None
        self.step_collector = StepCollector()
        self.window_collector = WindowCollector(window_size)
        self.model = model
        if device is None:
            if self.model is not None:
                device = next(self.model.parameters()).device
        else:
            if self.model is not None:
                self.model.to(device)

        self.device = device

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

    def reinit(self):
        self.step_collector.reinit()
        self.window_collector.reinit()

    def begin_train(self):
        self.reinit()
        self.on_train_begin()

    def on_train_begin(self):
        pass

    @abstractmethod
    def train_step(self, batch, batch_idx=None, epoch_idx=None):
        pass

    def end_train(self):
        self.on_train_end()

    def on_train_end(self):
        pass

    @contextmanager
    def on_train(self):
        self.begin_train()
        try:
            yield self
        finally:
            self.end_train()

    def begin_val(self):
        self.reinit()
        self.on_val_begin()

    def on_val_begin(self):
        pass

    def val_step(self, batch, batch_idx, epoch_idx):
        pass

    def end_val(self):
        self.on_val_end()

    def on_val_end(self):
        pass

    @contextmanager
    def on_val(self):
        with torch.no_grad():
            self._init()
            self.begin_val()
            try:
                yield self
            finally:
                self.end_val()

    def log(self, name=None, value=None):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log(k, v)
        else:
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.step_collector.add_value(name, value)
            self.window_collector.add_value(name, value)

    def get_batch_log(self):
        return self.window_collector.get_summary()

    def get_epoch_log(self):
        return self.step_collector.get_summary()

    def get_state_dict(self):
        pass
