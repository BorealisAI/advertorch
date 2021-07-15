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

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful, rand_init_delta
from advertorch.utils import clamp as batch_clamp

from torch import Tensor as t
from tqdm import tqdm

from .compute_fcts import l2_proj_maker, linf_proj_maker

#TODO:
#This is not blackbox attack, but random attack?
#Or, is there a random attack that inherits from blackbox
#attack?
#Should we have maximum queries per dataset, or per sample?
#Are these something we should enforce?
class BlackBoxAttack(Attack, LabelMixin):
    """
    Implements the base class for black-box attacks

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.

    Attributes:
        statement: gremlin adversarial robustness statement
        smooth_tree: Using smooth_tree is True, otherwise uses original tree
        max_loss_queries: max number of calls to model per data point
        max_crit_queries: max number of calls to early stopping criterion  per data poinr
        eps: perturbation limit according to lp-ball
        lr: adversarial perturbation at each steps
        p: norm for the lp-ball constraint
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.5,
            clip_min=0., clip_max=1.,
            max_loss_queries=np.inf,
            max_crit_queries=np.inf,
            lr=0.01, #todo: rename eps_iter
            p="inf",
            targeted=False
        ):

        super().__init__(predict, loss_fn, clip_min, clip_max)

        #TODO: rename this to "ord"
        assert p in ["inf", "2"], "L-{} is not supported".format(p)
        assert not (np.isinf(max_loss_queries) and np.isinf(max_crit_queries)), "one of the budgets has to be finite!"

        self.eps = eps
        self.lr = lr
        self.p = p
        self.max_loss_queries = max_loss_queries
        self.max_crit_queries = max_crit_queries

        self.targeted = targeted

        # the _proj method takes pts and project them into the constraint set:
        # which are
        #  1. eps lp-ball around xs
        #  2. valid data pt range [lb, ub]
        # it is meant to be used within `self._run` and `self._suggest`
        self._proj = None
        # a handy flag for _suggest method to denote whether the provided xs is a
        # new batch (i.e. the first iteration within `self.run`)
        self.is_new_batch = False

    def _suggest(self, xs_t, loss_fct):
        """
        .. Note::
            this method is assumed to be used only within _run, one implication is that
            `self._proj` is redefined every time `self.run` is called which might be used
            by `self._suggest`

        Args:
            xs_t: batch_size x dim x (torch tensor)
            loss_fct: function to query (the attacker would like to maximize) (batch_size data pts -> R^{batch_size}

        Returns:
            suggested xs as a (torch tensor)and the used number of queries per data point
                i.e. a tuple of (batch_size x dim x .. tensor, batch_size array of number queries used)
        """
        raise NotImplementedError

    def _proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        # replace xs only if not done
        xs_t = sugg_xs_t * (1.0 - dones_mask_t) + xs_t * dones_mask_t
        return xs_t

    def _loss_fct(self, x, y):
        x_ = torch.FloatTensor(x).to(self.device)
        y_ = torch.LongTensor(y).to(self.device)
        loss = nn.CrossEntropyLoss(reduction="none")(self.predict(x_), y_)

        if self.targeted:
            loss = -loss
        
        return loss.cpu().numpy()

    def _early_stop_crit_fct(self, x, y):
        x_ = torch.FloatTensor(x).to(self.device)
        y_ = torch.LongTensor(y).to(self.device)
        done_mask = torch.argmax(self.predict(x_), dim=1) != y_
        return done_mask.cpu().numpy()

    def _run(self, xs_t, loss_fct, early_stop_crit_fct, lb, ub):
        """
        Attack with `xs` as data points. This will be called by `generate_counterexamples`.
        Inputs should be encoded (normalzied).

        Args:
            xs_t: data points to be perturbed adversarially (numpy array)
            loss_fct: loss function (m data pts -> R^m)
            early_stop_crit_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified
            lb: lower bound of each feature (column)
            ub: upper bound of each feature (column)

        Returns:
            dict of logs whose length is the number of iterations
            adversarial examples
            mask showing if attacks are successful
        """

        # convert to tensor
        batch_size = xs_t.shape[0]
        num_axes = len(xs_t.shape[1:])
        num_loss_queries = np.zeros(batch_size)
        num_crit_queries = np.zeros(batch_size)

        dones_mask = early_stop_crit_fct(xs_t.cpu().numpy())
        correct_classified_mask = np.logical_not(dones_mask)

        # list of logs to be returned
        logs_dict = {
            "total_loss": [],
            "total_successes": [],
            "total_failures": [],
            "iteration": [],
            "total_loss_queries": [],
            "total_crit_queries": [],
            "num_loss_queries_per_iteration": [],
            "num_crit_queries_per_iteration": [],
        }

        # ignore this batch of xs if all are misclassified
        if sum(correct_classified_mask) == 0:
            return logs_dict

        # init losses and cosine similarity for performance tracking
        losses = np.zeros(batch_size)

        # make a projector into xs lp-ball and within valid pixel range
        #TODO : I don't like how this defines a new function within a batch
        if self.p == "2":
            _proj = l2_proj_maker(xs_t, self.eps)
            self._proj = lambda new_x: batch_clamp(_proj(new_x), lb, ub)
        elif self.p == "inf":
            _proj = linf_proj_maker(xs_t, self.eps)
            self._proj = lambda new_x: batch_clamp(_proj(new_x), lb, ub)
        else:
            raise Exception("Undefined l-p!")

        # iterate till model evasion or budget exhaustion
        # to inform self._suggest this is  a new batch
        self.is_new_batch = True
        its = 0
        while True:
            if np.any(num_loss_queries >= self.max_loss_queries):
                print("#loss queries exceeded budget, exiting")
                break
            if np.any(num_crit_queries >= self.max_crit_queries):
                print("#crit_queries exceeded budget, exiting")
                break
            if np.all(dones_mask):
                print("all data pts are misclassified, exiting")
                break
            # propose new perturbations
            sugg_xs_t, num_loss_queries_per_step = self._suggest(xs_t, loss_fct)
            # project around xs and within pixel range and
            # replace xs only if not done
            dones_mask_t = t(dones_mask.reshape(-1, *[1] * num_axes).astype(np.float32)).to(self.device)
            xs_t = self._proj_replace(xs_t, sugg_xs_t, dones_mask_t)
            # update number of queries (note this is done before updating dones_mask)
            num_loss_queries += num_loss_queries_per_step * (1.0 - dones_mask)
            num_crit_queries += 1.0 - dones_mask
            losses = loss_fct(xs_t.cpu().numpy()) * (1.0 - dones_mask) + losses * dones_mask
            # update dones mask
            dones_mask = np.logical_or(dones_mask, early_stop_crit_fct(xs_t.cpu().numpy()))
            its += 1
            self.is_new_batch = False

            # update logs
            logs_dict["total_loss"].append(sum(losses))
            logs_dict["total_successes"].append(sum(dones_mask * correct_classified_mask))
            logs_dict["total_failures"].append(sum(np.logical_not(dones_mask) * correct_classified_mask))
            logs_dict["iteration"].append(its)
            # assuming all data pts consume the same number of queries per step
            logs_dict["num_loss_queries_per_iteration"].append(num_loss_queries_per_step[0])
            logs_dict["num_crit_queries_per_iteration"].append(1)
            logs_dict["total_loss_queries"].append(sum(num_loss_queries * dones_mask * correct_classified_mask))
            logs_dict["total_crit_queries"].append(sum(num_crit_queries * dones_mask * correct_classified_mask))

        success_mask = dones_mask * correct_classified_mask

        # set self._proj to None to ensure it is intended use??
        self._proj = None

        return logs_dict, xs_t, dones_mask

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        .. Note::
            Mask output is before projecting the adversarial examples,
            i.e., mapping back to integers.
            Therefore, the actual mask could be different from the returning
            one. This mask could be helpful to understand the reason of
            unsuccessful attacks.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.

        :returns: tuple (logs_dict, xs_t, dones_mask) 
            WHERE
                dict logs_dict contains log information
                np.ndarray xs_t contains adversarial examples
                np.ndarray dones_mask indicates if attack was successful
        """
        x, y = self._verify_and_process_inputs(x, y)

        loss_fct = partial(self._loss_fct, y=y.cpu().numpy())
        early_stop_crit_fct = partial(self._early_stop_crit_fct, y=y.cpu().numpy())
        self.device = y.device

        if torch.is_tensor(self.eps):
            if len(self.eps) != len(y):
                raise ValueError("eps should have the same length as x.")
            self.eps = self.eps
        elif isinstance(self.eps, float):
            self.eps = torch.Tensor([self.eps] * len(y)).to(self.device)
        else:
            raise ValueError("eps should be float or Tensor.")

        #lb, ub = self._get_clip_bounds(X, self.eps)
        if isinstance(self.clip_min, float):
            clip_min = self.clip_min * torch.ones_like(x)
        elif isinstance(self.clip_min, (np.ndarray, list)):
            clip_min = torch.FloatTensor(self.clip_min).to(x.device)  # type: ignore
        elif isinstance(self.clip_min, torch.Tensor):
            clip_min = self.clip_min.to(x.device)  # type: ignore

        else:
            raise ValueError("Unknown clip_min format.")

        if isinstance(self.clip_max, float):
            clip_max = self.clip_max * torch.ones_like(x)
        elif isinstance(self.clip_max, (np.ndarray, list)):
            clip_max = torch.FloatTensor(self.clip_max).to(x.device)  # type: ignore
        elif isinstance(self.clip_max, torch.Tensor):
            clip_max = self.clip_max.to(x.device)  # type: ignore

        else:
            raise ValueError("Unknown clip_max format.")

        return self._run(
            x, loss_fct, early_stop_crit_fct, clip_min, clip_max
        )