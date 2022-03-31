# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from math import inf
from operator import mul
from functools import reduce

import numpy as np

import torch

from advertorch.utils import batch_clamp


def pytorch_wrapper(func):
    def wrapped_func(x):
        x_numpy = x.cpu().data.numpy()
        output = func(x_numpy)
        output = torch.from_numpy(output)
        output = output.to(x.device)

        return output

    return wrapped_func


def _check_param(param, x, name):
    if isinstance(param, (bool, int, float)):
        new_param = param * torch.ones_like(x)
    elif isinstance(param, (np.ndarray, list)):
        if param.ndim != x.ndim:
            raise ValueError(f"Mismatched number of dimensions for {name}."
                             "  Expand dimensions to match input."
                             )
        new_param = torch.FloatTensor(param).to(x.device)  # type: ignore
    elif isinstance(param, torch.Tensor):
        if param.ndim != x.ndim:
            raise ValueError(f"Mismatched number of dimensions for {name}."
                             "  Expand dimensions to match input."
                             )
        new_param = param.to(x.device)  # type: ignore
    else:
        raise ValueError(f"Unknown format for {name}")

    return new_param


def _flatten(x):
    shape = x.shape
    if x.dim() == 2:
        flat_x = x
    else:
        flat_size = reduce(mul, shape[1:])
        flat_x = x.reshape(x.shape[0], flat_size)

    return shape, flat_x


def sample_clamp(x, clip_min, clip_max):
    new_x = torch.maximum(x, clip_min)
    new_x = torch.minimum(new_x, clip_max)
    return new_x


def _make_projector(eps, order, x, clip_min, clip_max):
    if order == inf:
        def proj(delta):
            delta = batch_clamp(eps, delta)
            delta = sample_clamp(
                x[:, None, :] + delta,
                clip_min[:, None, :],
                clip_max[:, None, :]
            ) - x[:, None, :]
            return delta
    else:
        def proj(delta):
            # find the samples that exceed the bounds
            # and project them back inside
            norm = torch.norm(delta, p=order, dim=-1)
            mask = (norm > eps[:, None]).float()  # out of bounds
            factor = torch.min(eps[:, None] / norm, torch.ones_like(norm))
            delta_norm = delta * factor[:, :, None]
            delta = mask[:, :, None] * delta_norm + \
                (1 - mask)[:, :, None] * delta

            delta = sample_clamp(
                x[:, None, :] + delta,
                clip_min[:, None, :],
                clip_max[:, None, :]
            ) - x[:, None, :]
            return delta

    return proj
