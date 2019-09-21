# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from .base import Attack
from .base import LabelMixin
from .utils import MarginalLoss
from ..utils import is_float_or_torch_tensor

__all__ = ['LinfSPSAAttack', 'spsa_grad', 'spsa_perturb']


def clamp_(dx, x, eps, clip_min, clip_max):
    dx_clamped = torch.clamp(dx, -eps, eps)
    x_adv = torch.clamp(x + dx_clamped, clip_min, clip_max)
    dx += x_adv - x - dx
    return dx


@torch.no_grad()
def spsa_grad(predict, loss_fn, x, y, v, delta):
    xshape = x.shape
    x = x.view(-1, *x.shape[2:])
    y = y.view(-1, *y.shape[2:])
    v = v.view(-1, *v.shape[2:])

    f = lambda xvar, yvar: loss_fn(predict(xvar), yvar)
    # assumes v != 0
    grad = (f(x + delta * v, y) - f(x - delta * v, y)) / (2 * delta * v)
    
    grad = grad.view(*xshape).mean(dim=0, keepdim=True)

    return grad


def spsa_perturb(predict, loss_fn, x, y, eps, delta, lr, nb_iter,
                 nb_sample, clip_min=0.0, clip_max=1.0):
    """
    """
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    dx = torch.zeros_like(x)
    x_ = x.expand(nb_sample, *x.shape[1:])
    y_ = y.expand(nb_sample, *y.shape[1:])
    v_ = torch.empty_like(x_)
    optimizer = torch.optim.Adam([dx], lr=lr)

    for ii in range(nb_iter):
        optimizer.zero_grad()
        v_ = v_.bernoulli_()
        v_ *= 2.0
        v_ -= 1.0
        grad = spsa_grad(predict, loss_fn, x_ + dx, y_, v_, delta)
        dx.grad = grad
        optimizer.step()
        dx = clamp_(dx, x, eps, clip_min, clip_max)
    
    x_adv = (x + dx).squeeze(0)
    
    return x_adv


class LinfSPSAAttack(Attack, LabelMixin):

    def __init__(self, predict, eps, delta=0.01, lr=0.01, nb_iter=1,
                 nb_sample=128, targeted=False, loss_fn=None,
                 clip_min=0.0, clip_max=1.0):

        if loss_fn is None:
            loss_fn = MarginalLoss(reduction="sum")
        super(LinfSPSAAttack, self).__init__(predict, loss_fn, clip_min, clip_max)

        assert is_float_or_torch_tensor(eps)
        assert is_float_or_torch_tensor(delta)
        assert is_float_or_torch_tensor(lr)

        self.eps = float(eps)
        self.delta = float(delta)
        self.lr = float(lr)
        self.nb_iter = int(nb_iter)
        self.nb_sample = int(nb_sample)
        self.targeted = bool(targeted)

    def perturb(self, x, y=None):  # pylint: disable=arguments-differ
        x, y = self._verify_and_process_inputs(x, y)

        if self.targeted:
            loss = self.loss_fn
        else:
            loss = lambda *args: -self.loss_fn(*args)
        
        return spsa_perturb(self.predict, loss, x, y, self.eps, self.delta,
                            self.lr, self.nb_iter, self.nb_sample,
                            self.clip_min, self.clip_max)
