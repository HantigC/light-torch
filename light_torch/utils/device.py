from typing import Union
import torch
import numpy as np


def to_numpy_cpu(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x
