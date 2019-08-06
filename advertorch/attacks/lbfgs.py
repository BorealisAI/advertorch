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
import torch
import torch.nn.functional as F

from advertorch.utils import calc_l2distsq

from .base import Attack
from .base import LabelMixin


L2DIST_UPPER = 1e10
COEFF_UPPER = 1e10
INVALID_LABEL = -1
UPPER_CHECK = 1e9


class LBFGSAttack(Attack, LabelMixin):
    """
    The attack that uses L-BFGS to minimize the distance of the original
    and perturbed images

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param batch_size: number of samples in the batch
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, num_classes, batch_size=1,
                 binary_search_steps=9, max_iterations=100,
                 initial_const=1e-2,
                 clip_min=0, clip_max=1, loss_fn=None, targeted=False):
        super(LBFGSAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        # XXX: should combine the input loss function with other things
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.targeted = targeted


    def _update_if_better(
            self, adv_img, labs, output, dist, batch_size,
            final_l2dists, final_labels, final_advs):
        for ii in range(batch_size):
            target_label = labs[ii]
            output_logits = output[ii]
            _, output_label = torch.max(output_logits, 0)
            di = dist[ii]
            if (di < final_l2dists[ii] and
                    output_label.item() == target_label):
                final_l2dists[ii] = di
                final_labels[ii] = output_label
                final_advs[ii] = adv_img[ii]


    def _update_loss_coeffs(
            self, labs, batch_size,
            loss_coeffs, coeff_upper_bound, coeff_lower_bound, output):
        for ii in range(batch_size):
            _, cur_label = torch.max(output[ii], 0)
            if cur_label.item() == int(labs[ii]):
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

        from scipy.optimize import fmin_l_bfgs_b

        def _loss_fn(adv_x_np, self, x, target, const):
            adv_x = torch.from_numpy(
                adv_x_np.reshape(x.shape)).float().to(
                x.device).requires_grad_()
            output = self.predict(adv_x)
            loss2 = torch.sum((x - adv_x) ** 2)
            loss_fn = F.cross_entropy(output, target, reduction='none')
            loss1 = torch.sum(const * loss_fn)
            loss = loss1 + loss2
            loss.backward()
            grad_ret = adv_x.grad.data.cpu().numpy().flatten().astype(float)
            loss = loss.data.cpu().numpy().flatten().astype(float)
            if not self.targeted:
                loss = -loss
            return loss, grad_ret

        x, y = self._verify_and_process_inputs(x, y)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * COEFF_UPPER
        loss_coeffs = x.new_ones(batch_size) * self.initial_const
        final_l2dists = [L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x.clone()
        clip_min = self.clip_min * np.ones(x.shape[:]).astype(float)
        clip_max = self.clip_max * np.ones(x.shape[:]).astype(float)
        clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

        for outer_step in range(self.binary_search_steps):
            init_guess = x.clone().cpu().numpy().flatten().astype(float)
            adv_x, f, _ = fmin_l_bfgs_b(_loss_fn,
                                        init_guess,
                                        args=(self, x.clone(), y, loss_coeffs),
                                        bounds=clip_bound,
                                        maxiter=self.max_iterations,
                                        iprint=0)

            adv_x = torch.from_numpy(
                adv_x.reshape(x.shape)).float().to(x.device)
            l2s = calc_l2distsq(x, adv_x)
            output = self.predict(adv_x)
            self._update_if_better(
                adv_x, y, output.data, l2s, batch_size,
                final_l2dists, final_labels, final_advs)
            self._update_loss_coeffs(
                y, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound,
                output.data)
        return final_advs
