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


class RandAttack(BlackBoxAttack):
    """
    Implements the binary based attack

    Random Attack: randomly flip signs till at the boundary

    Attributes:
        statement: gremlin adversarial robustness statement
        smooth_tree: uses smooth_tree if True, otherwise uses original tree predictions
        max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        eps: radius of lp-ball of perturbation
        lr: Not used
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

    def _suggest(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
        sgn_t = sign(ch.rand(_shape[0], dim) - 0.5)

        # perform the step
        new_xs = lp_step(self.xo_t, sgn_t.view(_shape), self.eps[:, None], self.p)
        return new_xs, np.ones(_shape[0])

