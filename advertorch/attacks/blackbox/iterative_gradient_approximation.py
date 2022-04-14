# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.attacks.utils import rand_init_delta

from advertorch.attacks.iterative_projected_gradient import LinfPGDAttack
from advertorch.attacks.iterative_projected_gradient import perturb_iterative

from .estimators import NESWrapper
from .utils import _flatten


class NESAttack(LinfPGDAttack):
    """
    Implements NES Attack https://arxiv.org/abs/1804.08598

    Employs Natural Evolutionary Strategies for Gradient Estimation.
    Generates Adversarial Examples using Projected Gradient Descent.

    Disclaimer: Computations are broadcasted, so it is advisable to use
    smaller batch sizes when nb_samples is large.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_samples: number of samples to use for gradient estimation
    :param fd_eta: step-size used for Finite Difference gradient estimation
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3,
            nb_samples=100, fd_eta=1e-2, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):

        super(NESAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted)

        self.nb_samples = nb_samples
        self.fd_eta = fd_eta

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
        shape, flat_x = _flatten(x)
        data_shape = tuple(shape[1:])

        def f(x):
            new_shape = (x.shape[0],) + data_shape
            input = x.reshape(new_shape)
            return self.predict(input)
        f_nes = NESWrapper(
            f, nb_samples=self.nb_samples, fd_eta=self.fd_eta
        )

        delta = torch.zeros_like(flat_x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, flat_x, self.ord, self.eps, self.clip_min, self.clip_max
            )
            delta.data = clamp(
                flat_x + delta.data, min=self.clip_min, max=self.clip_max
            ) - flat_x

        rval = perturb_iterative(
            flat_x, y, f_nes, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=None
        )

        return rval.data.reshape(shape)
