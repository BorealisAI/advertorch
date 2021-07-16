# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import torch

from .base import Attack
from .base import LabelMixin
from .utils import MarginalLoss
from ..utils import is_float_or_torch_tensor, batch_clamp, clamp

__all__ = ['LinfSPSAAttack', 'spsa_grad', 'spsa_perturb']


def linf_clamp_(dx, x, eps, clip_min, clip_max):
    """Clamps perturbation `dx` to fit L_inf norm and image bounds.

    Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
    to be in `[clip_min, clip_max]`.

    :param dx: perturbation to be clamped (inplace).
    :param x: the image.
    :param eps: maximum possible L_inf.
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.

    :return: the clamped perturbation `dx`.
    """

    dx_clamped = batch_clamp(eps, dx)
    x_adv = clamp(x + dx_clamped, clip_min, clip_max)
    # `dx` is changed *inplace* so the optimizer will keep
    # tracking it. the simplest mechanism for inplace was
    # adding the difference between the new value `x_adv - x`
    # and the old value `dx`.
    dx += x_adv - x - dx
    return dx


def _get_batch_sizes(n, max_batch_size):
    batches = [max_batch_size for _ in range(n // max_batch_size)]
    if n % max_batch_size > 0:
        batches.append(n % max_batch_size)
    return batches


@torch.no_grad()
def spsa_grad(predict, loss_fn, x, y, delta, nb_sample, max_batch_size):
    """Uses SPSA method to apprixmate gradient w.r.t `x`.

    Use the SPSA method to approximate the gradient of `loss_fn(predict(x), y)`
    with respect to `x`, based on the nonce `v`.

    :param predict: predict function (single argument: input).
    :param loss_fn: loss function (dual arguments: output, target).
    :param x: input argument for function `predict`.
    :param y: target argument for function `loss_fn`.
    :param v: perturbations of `x`.
    :param delta: scaling parameter of SPSA.
    :param reduction: how to reduce the gradients of the different samples.

    :return: return the approximated gradient of `loss_fn(predict(x), y)`
             with respect to `x`.
    """

    grad = torch.zeros_like(x)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    def f(xvar, yvar):
        return loss_fn(predict(xvar), yvar)
    x = x.expand(max_batch_size, *x.shape[1:]).contiguous()
    y = y.expand(max_batch_size, *y.shape[1:]).contiguous()
    v = torch.empty_like(x[:, :1, ...])

    for batch_size in _get_batch_sizes(nb_sample, max_batch_size):
        x_ = x[:batch_size]
        y_ = y[:batch_size]
        vb = v[:batch_size]
        vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
        v_ = vb.expand_as(x_).contiguous()
        x_shape = x_.shape
        x_ = x_.view(-1, *x.shape[2:])
        y_ = y_.view(-1, *y.shape[2:])
        v_ = v_.view(-1, *v.shape[2:])
        df = f(x_ + delta * v_, y_) - f(x_ - delta * v_, y_)
        df = df.view(-1, *[1 for _ in v_.shape[1:]])
        grad_ = df / (2. * delta * v_)
        grad_ = grad_.view(x_shape)
        grad_ = grad_.sum(dim=0, keepdim=False)
        grad += grad_
    grad /= nb_sample

    return grad


def spsa_perturb(predict, loss_fn, x, y, eps, delta, lr, nb_iter,
                 nb_sample, max_batch_size, clip_min=0.0, clip_max=1.0):
    """Perturbs the input `x` based on SPSA attack.

    :param predict: predict function (single argument: input).
    :param loss_fn: loss function (dual arguments: output, target).
    :param x: input argument for function `predict`.
    :param y: target argument for function `loss_fn`.
    :param eps: the L_inf budget of the attack.
    :param delta: scaling parameter of SPSA.
    :param lr: the learning rate of the `Adam` optimizer.
    :param nb_iter: number of iterations of the attack.
    :param nb_sample: number of samples for the SPSA gradient approximation.
    :param max_batch_size: maximum batch size to be evaluated at once.
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.

    :return: the perturbated input.
    """

    dx = torch.zeros_like(x)
    dx.grad = torch.zeros_like(dx)
    optimizer = torch.optim.Adam([dx], lr=lr)
    for _ in range(nb_iter):
        optimizer.zero_grad()
        dx.grad = spsa_grad(
            predict, loss_fn, x + dx, y, delta, nb_sample, max_batch_size)
        optimizer.step()
        dx = linf_clamp_(dx, x, eps, clip_min, clip_max)
    x_adv = x + dx

    return x_adv


class LinfSPSAAttack(Attack, LabelMixin):
    """SPSA Attack (Uesato et al. 2018).
    Based on: https://arxiv.org/abs/1802.05666

    :param predict: predict function (single argument: input).
    :param eps: the L_inf budget of the attack.
    :param delta: scaling parameter of SPSA.
    :param lr: the learning rate of the `Adam` optimizer.
    :param nb_iter: number of iterations of the attack.
    :param nb_sample: number of samples for SPSA gradient approximation.
    :param max_batch_size: maximum batch size to be evaluated at once.
    :param targeted: [description]
    :param loss_fn: loss function (dual arguments: output, target).
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.
    """

    def __init__(self, predict, eps, delta=0.01, lr=0.01, nb_iter=1,
                 nb_sample=128, max_batch_size=64, targeted=False,
                 loss_fn=None, clip_min=0.0, clip_max=1.0):

        if loss_fn is None:
            loss_fn = MarginalLoss(reduction="none")
        elif hasattr(loss_fn, "reduction") and \
                getattr(loss_fn, "reduction") != "none":
            warnings.warn("`loss_fn` is recommended to have "
                          "reduction='none' when used in SPSA attack")

        super(LinfSPSAAttack, self).__init__(predict, loss_fn,
                                             clip_min, clip_max)

        assert is_float_or_torch_tensor(eps)
        assert is_float_or_torch_tensor(delta)
        assert is_float_or_torch_tensor(lr)

        self.eps = float(eps)
        self.delta = float(delta)
        self.lr = float(lr)
        self.nb_iter = int(nb_iter)
        self.nb_sample = int(nb_sample)
        self.max_batch_size = int(max_batch_size)
        self.targeted = bool(targeted)

    def perturb(self, x, y=None):  # pylint: disable=arguments-differ
        """Perturbs the input `x` based on SPSA attack.

        :param x: input tensor.
        :param y: label tensor (default=`None`). if `self.targeted` is `False`,
                  `y` is the ground-truth label. if it's `None`, then `y` is
                  computed as the predicted label of `x`.
                  if `self.targeted` is `True`, `y` is the target label.

        :return: the perturbated input.
        """

        x, y = self._verify_and_process_inputs(x, y)

        if self.targeted:
            def loss_fn(*args):
                return self.loss_fn(*args)

        else:
            def loss_fn(*args):
                return -self.loss_fn(*args)

        return spsa_perturb(self.predict, loss_fn, x, y, self.eps, self.delta,
                            self.lr, self.nb_iter, self.nb_sample,
                            self.max_batch_size, self.clip_min, self.clip_max)
