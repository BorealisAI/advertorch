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
from advertorch.utils import clamp
from advertorch.utils import to_one_hot

from .base import Attack
from .base import LabelMixin
from .utils import is_successful

L2DIST_UPPER = 1e10
TARGET_MULT = 10000.0
INVALID_LABEL = -1


class SpatialTransformAttack(Attack, LabelMixin):
    """
    Spatially Transformed Attack (Xiao et al. 2018)
    https://openreview.net/forum?id=HyydRMZC-

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the adversarial examples.
    :param initial_const: initial value of the constant c
    :param max_iterations: the maximum number of iterations
    :param search_steps: number of search times to find the optimum
    :param loss_fn: loss function
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param targeted: if the attack is targeted
    """

    def __init__(self, predict, num_classes, confidence=0,
                 initial_const=1, max_iterations=1000,
                 search_steps=1, loss_fn=None,
                 clip_min=0.0, clip_max=1.0,
                 abort_early=True, targeted=False):
        super(SpatialTransformAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.num_classes = num_classes
        self.confidence = confidence
        self.initial_const = initial_const
        self.max_iterations = max_iterations
        self.search_steps = search_steps
        self.abort_early = abort_early
        self.targeted = targeted

    def _loss_fn_spatial(self, grid, x, y, const, grid_ori):
        imgs = x.clone()
        grid = torch.from_numpy(
            grid.reshape(grid_ori.shape)).float().to(
            x.device).requires_grad_()
        delta = grid_ori - grid

        adv_img = F.grid_sample(imgs, grid)
        output = self.predict(adv_img)
        real = (y * output).sum(dim=1)
        other = (
            (1.0 - y) * output - (y * TARGET_MULT)).max(1)[0]
        if self.targeted:
            loss1 = clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = clamp(real - other + self.confidence, min=0.)
        loss2 = self.initial_const * (
            torch.sqrt((((
                delta[:, :, 1:] - delta[:, :, :-1] + 1e-10) ** 2)).view(
                    delta.shape[0], -1).sum(1)) +
            torch.sqrt(((
                delta[:, 1:, :] - delta[:, :-1, :] + 1e-10) ** 2).view(
                    delta.shape[0], -1).sum(1)))
        loss = torch.sum(loss1) + torch.sum(loss2)
        loss.backward()
        grad_ret = grid.grad.data.cpu().numpy().flatten().astype(float)
        grid.grad.data.zero_()
        return loss.data.cpu().numpy().astype(float), grad_ret

    def _update_if_better(
            self, adv_img, labs, output, dist, batch_size,
            final_l2dists, final_labels, final_advs, step, final_step):

        for ii in range(batch_size):
            target_label = labs[ii]
            output_logits = output[ii]
            _, output_label = torch.max(output_logits, 0)
            di = dist[ii]
            if (di < final_l2dists[ii] and
                    is_successful(
                    int(output_label.item()), int(target_label),
                    self.targeted)):
                final_l2dists[ii] = di
                final_labels[ii] = output_label
                final_advs[ii] = adv_img[ii]
                final_step[ii] = step

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        batch_size = len(x)
        loss_coeffs = x.new_ones(batch_size) * self.initial_const
        final_l2dists = [L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_step = [INVALID_LABEL] * batch_size
        final_advs = torch.zeros_like(x)

        # TODO: refactor the theta generation
        theta = torch.tensor([[[1., 0., 0.],
                               [0., 1., 0.]]]).to(x.device)
        theta = theta.repeat((x.shape[0], 1, 1))


        grid = F.affine_grid(theta, x.size())

        grid_ori = grid.clone()
        y_onehot = to_one_hot(y, self.num_classes).float()

        clip_min = np.ones(grid_ori.shape[:]) * -1
        clip_max = np.ones(grid_ori.shape[:]) * 1
        clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))
        grid_ret = grid.clone().data.cpu().numpy().flatten().astype(float)
        from scipy.optimize import fmin_l_bfgs_b
        for outer_step in range(self.search_steps):
            grid_ret, f, d = fmin_l_bfgs_b(
                self._loss_fn_spatial,
                grid_ret,
                args=(
                    x.clone().detach(),
                    y_onehot, loss_coeffs,
                    grid_ori.clone().detach()),
                maxiter=self.max_iterations,
                bounds=clip_bound,
                iprint=0,
                maxls=100,
            )
            grid = torch.from_numpy(
                grid_ret.reshape(grid_ori.shape)).float().to(x.device)
            adv_x = F.grid_sample(x.clone(), grid)
            l2s = calc_l2distsq(grid.data, grid_ori.data)
            output = self.predict(adv_x)
            self._update_if_better(
                adv_x.data, y, output.data, l2s, batch_size,
                final_l2dists, final_labels, final_advs,
                outer_step, final_step)

        return final_advs
