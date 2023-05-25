from collections import defaultdict
import queue

import numpy as np


class WindowCollector:
    def __init__(self, window_size=None):
        self._windows = defaultdict(lambda: queue.deque(maxlen=window_size))

    def add_value(self, name, value):
        self._windows[name].append(value)

    def get_value(self, name, apply=np.mean):
        return apply(self._windows[name])

    def get_summary(self, apply=np.mean):
        return {k: apply(v) for k, v in self._windows.items()}


class StepCollector:
    def __init__(self):
        self._accumulators = defaultdict(int)
        self._amounts = defaultdict(int)

    def add_value(self, name, value):
        length = self._amounts[name]
        total = self._accumulators[name] * length + value
        self._amounts[name] += 1
        self._accumulators[name] = total / self._amounts[name]

    def get_value(self, name):
        return self._accumulators[name]

    def get_summary(self):
        return dict(self._accumulators)
