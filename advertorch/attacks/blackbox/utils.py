import numpy as np

import torch

def _check_param(param, x, name):
    if isinstance(param, float):
        new_param = param * torch.ones_like(x)
    elif isinstance(param, (np.ndarray, list)):
        new_param = torch.FloatTensor(param).to(x.device)  # type: ignore
    elif isinstance(param, torch.Tensor):
        new_param = param.to(x.device)  # type: ignore
    else:
        raise ValueError("Unknown format for {}".format(name))

    return new_param

def track_stats():
    pass

def partial_perturb():
    #limit_queries=False
    #batchify:
    #dones = zeros()
    #old_adv = zeros()
    #current = x[~dones]
    #current_adv = step(current, ...)
    #old_adv[~dones] = current_adv
    #dones = check(...)
    pass