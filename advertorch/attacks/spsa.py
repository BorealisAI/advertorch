# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

from boltons.iterutils import chunked_iter
import torch

from .base import Attack
from .base import LabelMixin
from .utils import MarginalLoss
from ..utils import is_float_or_torch_tensor

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

    dx_clamped = torch.clamp(dx, -eps, eps)
    x_adv = torch.clamp(x + dx_clamped, clip_min, clip_max)
    # `dx` is changed *inplace* so the optimizer will keep
    # tracking it. the simplest mechanism for inplace was
    # adding the difference between the new value `x_adv - x`
    # and the old value `dx`.
    dx += x_adv - x - dx
    return dx


@torch.no_grad()
def spsa_grad(predict, loss_fn, x, y, v, delta, reduction="mean"):
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
    assert reduction in (
        "mean", "sum"), "`reduction` should be eigther 'mean' or 'sum'"

    xshape = x.shape
    x = x.view(-1, *x.shape[2:])
    y = y.view(-1, *y.shape[2:])
    v = v.view(-1, *v.shape[2:])

    def f(xvar, yvar):
        return loss_fn(predict(xvar), yvar)

    # assumes v != 0
    df = f(x + delta * v, y) - f(x - delta * v, y)
    df = df.view(-1, *[1 for _ in v.shape[1:]])

    grad = df / (2 * delta * v)
    grad = grad.view(*xshape)
    if reduction == "mean":
        grad = grad.mean(dim=0, keepdim=True)
    elif reduction == "sum":
        grad = grad.sum(dim=0, keepdim=True)
    else:
        raise ValueError("`reduction` should be eigther 'mean' or 'sum'")

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

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    dx = torch.zeros_like(x) # is it necessary to define x after the expansin of x?
    dx.grad = torch.zeros_like(dx)
    optimizer = torch.optim.Adam([dx], lr=lr)
    # Todo: put the following logic in spsa_grad.
    xb = x.expand(max_batch_size, *x.shape[1:]).contiguous()
    yb = y.expand(max_batch_size, *y.shape[1:]).contiguous()
    vb = torch.empty_like(xb[:, 0:1, ...])
    for _ in range(nb_iter):
        optimizer.zero_grad()
        for nb_sample_per_batch in chunked_iter(range(nb_sample), max_batch_size): # the use of chunked_iter, and following len() feels a bit clumsy. Any better alternative?
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            x_ = xb[:len(nb_sample_per_batch)]
            y_ = yb[:len(nb_sample_per_batch)]
            v_ = vb[:len(nb_sample_per_batch)].expand_as(x_).contiguous()
            grad = spsa_grad(predict, loss_fn, x_ + dx, y_,
                    v_, delta, reduction="sum")
            dx.grad += grad
        dx.grad /= nb_sample
        optimizer.step()
        dx = linf_clamp_(dx, x, eps, clip_min, clip_max)

    x_adv = (x + dx).squeeze(0)

    return x_adv


class LinfSPSAAttack(Attack, LabelMixin):

    def __init__(self, predict, eps, delta=0.01, lr=0.01, nb_iter=1,
                 nb_sample=128, max_batch_size=64, targeted=False,
                 loss_fn=None, clip_min=0.0, clip_max=1.0):
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

        if loss_fn is None:
            loss_fn = MarginalLoss(reduction="none")
        elif hasattr(loss_fn, "reduction") and \
                getattr(loss_fn, "reduction") != "none":
            warnings.warn("`loss_fn` is recommended to have "
                          "eduction='none' when used in SPSA attack")

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
