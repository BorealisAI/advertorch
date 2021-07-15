# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import torch as ch
from torch import Tensor as t

from .base import BlackBoxAttack
from .compute_fcts import lp_step


class NESAttack(BlackBoxAttack):
    """
    NES Attack: https://arxiv.org/pdf/1804.08598.pdf

    Attributes:
        statement: gremlin adversarial robustness statement
        smooth_tree: uses smooth_tree if True, otherwise uses original tree predictions
        max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        eps: radius of lp-ball of perturbation
        p: specifies lp-norm  of perturbation
        fd_eta: forward difference step
        lr: learning rate of each step
        q: number of noise samples per NES step
    """

    def __init__(
            self, predict, fd_eta, 
            q, loss_fn=None, eps=0.5,
            clip_min=0., clip_max=1.,
            max_loss_queries=np.inf,
            max_crit_queries=np.inf,
            lr=0.01, #todo: rename eps_iter
            p="inf", targeted=False
        ):

        super().__init__(
            predict=predict, 
            loss_fn=loss_fn, 
            eps=eps,
            clip_min=clip_min, 
            clip_max=clip_max,
            max_loss_queries=max_loss_queries,
            max_crit_queries=max_crit_queries,
            lr=lr, #todo: rename eps_iter
            p=p,
            targeted=targeted
        )

        self.q = q
        self.fd_eta = fd_eta

    def _suggest(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = ch.zeros_like(xs_t)
        for i in range(self.q):
            exp_noise = ch.randn_like(xs_t) / (dim ** 0.5)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t - self.fd_eta * exp_noise
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / (2.0 * self.fd_eta)
            gs_t += t(est_deriv.reshape(-1, *[1] * num_axes)).to(xs_t.device) * exp_noise

        # perform the step
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        return new_xs, 2 * self.q * np.ones(_shape[0])
