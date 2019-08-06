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
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin
from .utils import is_successful


class SinglePixelAttack(Attack, LabelMixin):
    """
    Single Pixel Attack
    Algorithm 1 in https://arxiv.org/pdf/1612.06299.pdf

    :param predict: forward pass function.
    :param max_pixels: max number of pixels to perturb.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, max_pixels=100, clip_min=0.,
                 loss_fn=None, clip_max=1., comply_with_foolbox=False,
                 targeted=False):
        super(SinglePixelAttack, self).__init__(
            predict=predict, loss_fn=None,
            clip_min=clip_min, clip_max=clip_max)
        self.max_pixels = max_pixels
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.comply_with_foolbox = comply_with_foolbox
        self.targeted = targeted


    def perturb_single(self, x, y):
        # x shape [C * H * W]
        if self.comply_with_foolbox is True:
            np.random.seed(233333)
            rand_np = np.random.permutation(x.shape[1] * x.shape[2])
            pixels = torch.from_numpy(rand_np)
        else:
            pixels = torch.randperm(x.shape[1] * x.shape[2])
        pixels = pixels.to(x.device)
        pixels = pixels[:self.max_pixels]
        for ii in range(self.max_pixels):
            row = pixels[ii] % x.shape[2]
            col = pixels[ii] // x.shape[2]
            for val in [self.clip_min, self.clip_max]:
                adv = replicate_input(x)
                for mm in range(x.shape[0]):
                    adv[mm, row, col] = val
                out_label = self._get_predicted_label(adv.unsqueeze(0))
                if self.targeted is True:
                    if int(out_label[0]) == int(y):
                        return adv
                else:
                    if int(out_label[0]) != int(y):
                        return adv
        return x

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        return _perturb_batch(self.perturb_single, x, y)


class LocalSearchAttack(Attack, LabelMixin):
    """
    Local Search Attack
    Algorithm 3 in https://arxiv.org/pdf/1612.06299.pdf

    :param predict: forward pass function.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param p: parameter controls pixel complexity
    :param r: perturbation value
    :param loss_fn: loss function
    :param d: the half side length of the neighbourhood square
    :param t: the number of pixels perturbed at each round
    :param k: the threshold for k-misclassification
    :param round_ub: an upper bound on the number of rounds
    """

    def __init__(self, predict, clip_min=0., clip_max=1., p=1., r=1.5,
                 loss_fn=None, d=5, t=5, k=1, round_ub=10, seed_ratio=0.1,
                 max_nb_seeds=128, comply_with_foolbox=False, targeted=False):
        super(LocalSearchAttack, self).__init__(
            predict=predict, clip_max=clip_max,
            clip_min=clip_min, loss_fn=None)
        self.p = p
        self.r = r
        self.d = d
        self.t = t
        self.k = k
        self.round_ub = round_ub
        self.seed_ratio = seed_ratio
        self.max_nb_seeds = max_nb_seeds
        self.comply_with_foolbox = comply_with_foolbox
        self.targeted = targeted

        if clip_min is None or clip_max is None:
            raise ValueError("{} {}".format(
                LocalSearchAttack,
                "must have clip_min and clip_max specified as scalar values."))

    def perturb_single(self, x, y):
        # x shape C * H * W
        rescaled_x = replicate_input(x)
        best_img = None
        best_dist = np.inf
        rescaled_x, lb, ub = self._rescale_to_m0d5_to_0d5(
            rescaled_x, vmin=self.clip_min, vmax=self.clip_max)

        if self.comply_with_foolbox is True:
            np.random.seed(233333)
            init_rand = np.random.permutation(x.shape[1] * x.shape[2])
        else:
            init_rand = None

        # Algorithm 3 in v1

        pxy = self._random_sample_seeds(
            x.shape[1], x.shape[2], seed_ratio=self.seed_ratio,
            max_nb_seeds=self.max_nb_seeds, init_rand=init_rand)
        pxy = pxy.to(x.device)
        ii = 0
        if self.comply_with_foolbox:
            adv = rescaled_x
        while ii < self.round_ub:
            if not self.comply_with_foolbox:
                adv = replicate_input(rescaled_x)
            # Computing the function g using the neighbourhood
            if self.comply_with_foolbox:
                rand_np = np.random.permutation(len(pxy))[:self.max_nb_seeds]
                pxy = pxy[torch.from_numpy(rand_np)]
            else:
                pxy = pxy[torch.randperm(len(pxy))[:self.max_nb_seeds]]

            pert_lst = [
                self._perturb_seed_pixel(
                    adv, self.p, int(row), int(col)) for row, col in pxy]
            # Compute the score for each pert in the list
            scores, curr_best_img, curr_best_dist = self._rescale_x_score(
                self.predict, pert_lst, y, x, best_dist)
            if curr_best_img is not None:
                best_img = curr_best_img
                best_dist = curr_best_dist
            _, indices = torch.sort(scores)
            indices = indices[:self.t]
            pxy_star = pxy[indices.data.cpu()]
            # Generation of the perturbed image adv
            for row, col in pxy_star:
                for b in range(x.shape[0]):
                    adv[b, int(row), int(col)] = self._cyclic(
                        self.r, lb, ub, adv[b, int(row), int(col)])
            # Check whether the perturbed image is an adversarial image
            revert_adv = self._revert_rescale(adv)
            curr_lb = self._get_predicted_label(revert_adv.unsqueeze(0))
            curr_dist = torch.sum((x - revert_adv) ** 2)
            if (is_successful(int(curr_lb), y, self.targeted) and
                    curr_dist < best_dist):
                best_img = revert_adv
                best_dist = curr_dist
                return best_img
            elif is_successful(curr_lb, y, self.targeted):
                return best_img
            pxy = [
                (row, col)
                for rowcenter, colcenter in pxy_star
                for row in range(
                    int(rowcenter) - self.d, int(rowcenter) + self.d + 1)
                for col in range(
                    int(colcenter) - self.d, int(colcenter) + self.d + 1)]
            pxy = list(set((row, col) for row, col in pxy if (
                0 <= row < x.shape[2] and 0 <= col < x.shape[1])))
            pxy = torch.FloatTensor(pxy)
            ii += 1
        if best_img is None:
            return x
        return best_img

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        return _perturb_batch(self.perturb_single, x, y)

    def _rescale_to_m0d5_to_0d5(self, x, vmin=0., vmax=1.):
        x = x - (vmin + vmax) / 2
        x = x / (vmax - vmin)
        return x, -0.5, 0.5


    def _revert_rescale(self, x, vmin=0., vmax=1.):
        x_revert = x.clone()
        x_revert = x_revert * (vmax - vmin)
        x_revert = x_revert + (vmin + vmax) / 2
        return x_revert


    def _random_sample_seeds(self, h, w, seed_ratio, max_nb_seeds, init_rand):
        n = int(seed_ratio * h * w)
        n = min(n, max_nb_seeds)
        if init_rand is not None:
            locations = torch.from_numpy(init_rand)[:n]
        else:
            locations = torch.randperm(h * w)[:n]
        p_x = locations.int() % w
        p_y = locations.int() / w
        pxy = list(zip(p_x, p_y))
        pxy = torch.Tensor(pxy)
        return pxy


    def _perturb_seed_pixel(self, x, p, row, col):
        x_pert = replicate_input(x)
        for ii in range(x.shape[0]):
            if x[ii, row, col] > 0:
                x_pert[ii, row, col] = p
            elif x[ii, row, col] < 0:
                x_pert[ii, row, col] = -1 * p
            else:
                x_pert[ii, row, col] = 0
        return x_pert


    def _cyclic(self, r, lower_bound, upper_bound, i_bxy):
        # Algorithm 2 in v1
        result = r * i_bxy
        if result < lower_bound:
            result = result + (upper_bound - lower_bound)
        elif result > upper_bound:
            result = result - (upper_bound - lower_bound)
        return result


    def _rescale_x_score(self, predict, x, y, ori, best_dist):
        x = torch.stack(x)
        x = self._revert_rescale(x)

        batch_logits = predict(x)
        scores = nn.Softmax(dim=1)(batch_logits)[:, y]

        if not self.comply_with_foolbox:
            x = clamp(x, self.clip_min, self.clip_max)
            batch_logits = predict(x)

        _, bests = torch.max(batch_logits, dim=1)
        best_img = None
        for ii in range(len(bests)):
            curr_dist = torch.sum((x[ii] - ori) ** 2)
            if (is_successful(
                    int(bests[ii]), y, self.targeted) and
                    curr_dist < best_dist):
                best_img = x[ii]
                best_dist = curr_dist
        scores = nn.Softmax(dim=1)(batch_logits)[:, y]
        return scores, best_img, best_dist


def _perturb_batch(perturb_single, x, y):
    for ii in range(len(x)):
        temp = perturb_single(x[ii], y[ii])[None, :, :, :]
        if ii == 0:
            result = temp
        else:
            result = torch.cat((result, temp))
    return result
