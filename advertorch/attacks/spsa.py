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

__all__ = ['SPSALinfAttack', 'spsa_grad', 'spsa_perturb']


def clamp_(dx, x, eps, clip_min, clip_max):
    dx_clamped = torch.clamp(dx, -eps, eps)
    x_adv = torch.clamp(x + dx_clamped, clip_min, clip_max)
    dx += x_adv - x - dx
    return dx


@torch.no_grad()
def spsa_grad(predict, loss_fn, x, y, v, delta):
    xshape = x.shape
    x = x.view(-1, x.shape[1:])
    y = y.view(-1, y.shape[1:])
    v = v.view(-1, v.shape[1:])

    f = lambda xvar, yvar: loss_fn(predict(xvar), yvar)
    # assumes v != 0
    grad = f(x + delta * v, y) - f(x - delta * v, y) / (2 * delta * v)
    
    grad = grad.view(*xshape).mean(dim=0, keepdim=True)

    return grad


def spsa_perturb(predict, loss_fn, x, y, eps, delta, lr, nb_iter,
                 nb_sample, clip_min=0.0, clip_max=1.0):
    """
    """
    dx = x.new_zeros((1, *x.shape))
    x = x.unsqueeze(0).expand(nb_sample, *x.shape)
    y = y.unsqueeze(0).expand(nb_sample, *y.shape)
    v = torch.empty_like(x)
    optimizer = torch.optim.SGD([dx], lr=lr)

    for _ in range(nb_iter):
        optimizer.zero_grad()
        v = v.bernoulli_()
        v *= 2.0
        v -= 1.0
        grad = spsa_grad(predict, loss_fn, x, y, v, delta)
        dx.grad = grad.sign()
        optimizer.step()
        dx = clamp_(dx, x, eps, clip_min, clip_max)

    x_adv = x[0] + dx[0]
    
    return x_adv


class SPSALinfAttack(Attack, LabelMixin):

    def __init__(self, predict, eps, delta=0.01, lr=0.01, nb_iter=1,
                 nb_sample=128, targeted=False, loss_fn=None,
                 clip_min=0.0, clip_max=1.0):

        if loss_fn is None:
            loss_fn = MarginalLoss(reduction="sum")
        super(SPSALinfAttack, self).__init__(predict, loss_fn, clip_min, clip_max)

        assert is_float_or_torch_tensor(eps)
        assert is_float_or_torch_tensor(delta)
        assert is_float_or_torch_tensor(lr)

        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.targeted = targeted

    def perturb(self, x, y=None):  # pylint: disable=arguments-differ
        x, y = self._verify_and_process_inputs(x, y)

        if self.targeted:
            loss = self.loss_fn
        else:
            loss = lambda *args: -self.loss_fn(*args)
        
        return spsa_perturb(self.predict, loss, x, y, self.eps, self.delta,
                            self.lr, self.nb_iter, self.nb_sample,
                            self.clip_min, self.clip_max)
    