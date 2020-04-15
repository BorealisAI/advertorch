# Copyright (c) 2019-present, Alexandre Araujo.
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

from advertorch.utils import calc_l2distsq
from advertorch.utils import calc_l1dist
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin
from .utils import is_successful


DIST_UPPER = 1e10
COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000
NUM_CHECKS = 10


class ElasticNetL1Attack(Attack, LabelMixin):
    """
    The ElasticNet L1 Attack, https://arxiv.org/abs/1709.04114

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param beta: hyperparameter trading off L2 minimization for L1 minimization
    :param decision_rule: EN or L1. Select final adversarial example from
                          all successful examples based on the least
                          elastic-net or L1 distortion criterion.
    :param loss_fn: loss function
    """

    def __init__(self, predict, num_classes, confidence=0,
                 targeted=False, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=False, initial_const=1e-3,
                 clip_min=0., clip_max=1., beta=1e-2, decision_rule='EN',
                 loss_fn=None):
        """ElasticNet L1 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(ElasticNetL1Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        self.beta = beta
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted
        self.decision_rule = decision_rule


    def _loss_fn(self, output, y_onehot, l1dist, l2distsq, const, opt=False):

        real = (y_onehot * output).sum(dim=1)
        other = ((1.0 - y_onehot) * output -
                 (y_onehot * TARGET_MULT)).max(1)[0]

        if self.targeted:
            loss_logits = clamp(other - real + self.confidence, min=0.)
        else:
            loss_logits = clamp(real - other + self.confidence, min=0.)
        loss_logits = torch.sum(const * loss_logits)

        loss_l2 = l2distsq.sum()

        if opt:
            loss = loss_logits + loss_l2
        else:
            loss_l1 = self.beta * l1dist.sum()
            loss = loss_logits + loss_l2 + loss_l1
        return loss


    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label
        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)).long(),
                       label] -= self.confidence
            else:
                output[torch.arange(len(label)).long(),
                       label] += self.confidence
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()

        return is_successful(pred, label, self.targeted)


    def _fast_iterative_shrinkage_thresholding(self, x, yy_k, xx_k):

        zt = self.global_step / (self.global_step + 3)

        upper = clamp(yy_k - self.beta, max=self.clip_max)
        lower = clamp(yy_k + self.beta, min=self.clip_min)

        diff = yy_k - x
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        xx_k_p_1 = (cond1 * upper) + (cond2 * x) + (cond3 * lower)
        yy_k.data = xx_k_p_1 + (zt * (xx_k_p_1 - xx_k))
        return yy_k, xx_k_p_1


    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, dist, batch_size,
            cur_dist, cur_labels,
            final_dist, final_labels, final_advs):

        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (dist < cur_dist) & self._is_successful(
            output_logits, target_label, True)

        cur_dist[mask] = dist[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (dist < final_dist) & self._is_successful(
            output_logits, target_label, True)
        final_dist[mask] = dist[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]


    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound):

        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10


    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const

        final_dist = [DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size

        final_advs = x.clone()
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_dist = torch.FloatTensor(final_dist).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):

            self.global_step = 0

            # slack vector from the paper
            yy_k = nn.Parameter(x.clone())
            xx_k = x.clone()

            cur_dist = [DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size

            cur_dist = torch.FloatTensor(cur_dist).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)

            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound

            lr = self.learning_rate

            for ii in range(self.max_iterations):

                # reset gradient
                if yy_k.grad is not None:
                    yy_k.grad.detach_()
                    yy_k.grad.zero_()

                # loss over yy_k with only L2 same as C&W
                # we don't update L1 loss with SGD because we use ISTA
                output = self.predict(yy_k)
                l2distsq = calc_l2distsq(yy_k, x)
                loss_opt = self._loss_fn(
                    output, y_onehot, None, l2distsq, loss_coeffs, opt=True)
                loss_opt.backward()

                # gradient step
                yy_k.data.add_(-lr, yy_k.grad.data)
                self.global_step += 1

                # ploynomial decay of learning rate
                lr = self.init_learning_rate * \
                    (1 - self.global_step / self.max_iterations)**0.5

                yy_k, xx_k = self._fast_iterative_shrinkage_thresholding(
                    x, yy_k, xx_k)

                # loss ElasticNet or L1 over xx_k
                with torch.no_grad():
                    output = self.predict(xx_k)
                    l2distsq = calc_l2distsq(xx_k, x)
                    l1dist = calc_l1dist(xx_k, x)

                    if self.decision_rule == 'EN':
                        dist = l2distsq + (l1dist * self.beta)
                    elif self.decision_rule == 'L1':
                        dist = l1dist
                    loss = self._loss_fn(
                        output, y_onehot, l1dist, l2distsq, loss_coeffs)

                    if self.abort_early:
                        if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                            if loss > prevloss * ONE_MINUS_EPS:
                                break
                            prevloss = loss

                    self._update_if_smaller_dist_succeed(
                        xx_k.data, y, output, dist, batch_size,
                        cur_dist, cur_labels,
                        final_dist, final_labels, final_advs)

            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)

        return final_advs
