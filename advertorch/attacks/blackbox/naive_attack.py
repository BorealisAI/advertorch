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

from .base import BlackBoxAttack
from .compute_fcts import lp_step, sign


class NaiveAttack(BlackBoxAttack):
    """
    Implements the binary based attack with sequential bit flipping

    Naive Attack: sequentially flip signs till at the boundary
    https://arxiv.org/pdf/1712.09491.pdf

     Attributes:
        statement: gremlin adversarial robustness statement
        smooth_tree: uses smooth_tree if True, otherwise uses original tree predictions
        max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        eps: radius of lp-ball of perturbation
        lr: learning rate of each step
        p: specifies lp-norm  of perturbation
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.5,
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

        self.xo_t = None
        self.sgn_t = None

        #TODO: I don't like state variables from runs
        #being saved in between runs.
        #I think this is done in between batches...
        #which suggests we need better batch handlers inside these
        #functions
        self.best_est_deriv = None
        self.i = 0

    def _suggest(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        add_queries = 0
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
            self.i = 0
        if self.i == 0:
            self.sgn_t = sign(ch.ones(_shape[0], dim))
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.lr, self.p)
            bxs_t = self.xo_t
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.lr
            self.best_est_deriv = est_deriv
            add_queries = 2
        self.sgn_t[:, self.i] *= -1
        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.lr, self.p)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.lr
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], self.i] *= -1.0
        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
            est_deriv < self.best_est_deriv
        ) * self.best_est_deriv

        # perform the step
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.lr, self.p)
        self.i += 1
        if self.i == dim:
            self.xo_t = new_xs.clone()
            self.i = 0
        return new_xs, np.ones(_shape[0]) + add_queries
