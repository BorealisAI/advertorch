# Copyright (c) 2019-present, Jérôme Rony.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim

from .base import Attack
from .base import LabelMixin


class DDNL2Attack(Attack, LabelMixin):
    """
    The decoupled direction and norm attack (Rony et al, 2018).
    Paper: https://arxiv.org/abs/1811.09600

    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param gamma: factor to modify the norm at each iteration.
    :param init_norm: initial norm of the perturbation.
    :param quantize: perform quantization at each iteration.
    :param levels: number of quantization levels (e.g. 256 for 8 bit images).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param loss_fn: loss function.
    """

    def __init__(
            self, predict, nb_iter=100, gamma=0.05, init_norm=1.,
            quantize=True, levels=256, clip_min=0., clip_max=1.,
            targeted=False, loss_fn=None):
        """
        Decoupled Direction and Norm L2 Attack implementation in pytorch.
        """
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        super(DDNL2Attack, self).__init__(predict, loss_fn, clip_min, clip_max)

        self.nb_iter = nb_iter
        self.gamma = gamma
        self.init_norm = init_norm
        self.quantize = quantize
        self.levels = levels
        self.targeted = targeted

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        s = self.clip_max - self.clip_min
        multiplier = 1 if self.targeted else -1
        batch_size = x.shape[0]
        data_dims = (1,) * (x.dim() - 1)
        norm = torch.full((batch_size,), s * self.init_norm,
                          device=x.device, dtype=torch.float)
        worst_norm = torch.max(
            x - self.clip_min, self.clip_max - x).flatten(1).norm(p=2, dim=1)

        # setup variable and optimizer
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.nb_iter, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(x)

        for i in range(self.nb_iter):
            scheduler.step()

            l2 = delta.data.flatten(1).norm(p=2, dim=1)
            logits = self.predict(x + delta)
            pred_labels = logits.argmax(1)
            ce_loss = self.loss_fn(logits, y)
            loss = multiplier * ce_loss

            is_adv = (pred_labels == y) if self.targeted else (
                pred_labels != y)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()

            # renorming gradient
            grad_norms = s * delta.grad.flatten(1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, *data_dims))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)

            delta.data.mul_((norm / delta.data.flatten(1).norm(
                p=2, dim=1)).view(-1, *data_dims))
            delta.data.add_(x)
            if self.quantize:
                delta.data.sub_(self.clip_min).div_(s)
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
                delta.data.mul_(s).add_(self.clip_min)
            delta.data.clamp_(self.clip_min, self.clip_max).sub_(x)

        return x + best_delta
