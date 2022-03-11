import numpy as np

import torch
import torch.nn as nn

def _check_param(param, x, name):
    if isinstance(param, (bool, int, float)):
        new_param = param * torch.ones_like(x)
    elif isinstance(param, (np.ndarray, list)):
        new_param = torch.FloatTensor(param).to(x.device)  # type: ignore
    elif isinstance(param, torch.Tensor):
        new_param = param.to(x.device)  # type: ignore
    else:
        raise ValueError("Unknown format for {}".format(name))

    return new_param