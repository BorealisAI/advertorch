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

import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.utils import calc_l2distsq
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin
from .utils import is_successful


CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
EPS = 1e-6
NUM_CHECKS = 10

try:
    boolean_type = torch.bool
except AttributeError:
    # Old version, use torch.uint8
    boolean_type = torch.uint8

class CarliniWagnerL2Attack(Attack, LabelMixin):
    """
    The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644

    :param predict: forward pass function.
    :param num_classes: number of classes.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm.
    :param binary_search_steps: number of binary search times to find the
        optimum.
    :param max_iterations: the maximum number of iterations.
    :param abort_early: if set to true, abort early if getting stuck in local
        min.
    :param initial_const: initial value of the constant c.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function.
    """

    def __init__(self, predict, num_classes, confidence=0,
                 targeted=False, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1., loss_fn=None):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(CarliniWagnerL2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted

    def _loss_fn(self, output, y_onehot, l2distsq, const):
        # TODO: move this out of the class and make this the default loss_fn
        #   after having targeted tests implemented
        real = (y_onehot * output).sum(dim=1)

        # TODO: make loss modular, write a loss class
        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)
                 ).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = clamp(real - other + self.confidence, min=0.)
        loss2 = (l2distsq).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
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


    def _forward_and_update_delta(
            self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):

        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        output = self.predict(adv)
        l2distsq = calc_l2distsq(adv, transimgs_rescale)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, output.data, adv.data


    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, l2distsq, batch_size,
            cur_l2distsqs, cur_labels,
            final_l2distsqs, final_labels, final_advs):

        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (l2distsq < cur_l2distsqs) & self._is_successful(
            output_logits, target_label, True)

        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (l2distsq < final_l2distsqs) & self._is_successful(
            output_logits, target_label, True)
        final_l2distsqs[mask] = l2distsq[mask]
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
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, l2distsq, output, adv_img = \
                    self._forward_and_update_delta(
                        optimizer, x_atanh, delta, y_onehot, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                self._update_if_smaller_dist_succeed(
                    adv_img, y, output, l2distsq, batch_size,
                    cur_l2distsqs, cur_labels,
                    final_l2distsqs, final_labels, final_advs)

            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)

        return final_advs

class CarliniWagnerLinfAttack(Attack, LabelMixin):
    """
    The Carlini and Wagner LInfinity Attack, https://arxiv.org/abs/1608.04644

    :param predict: forward pass function (pre-softmax).
    :param num_classes: number of classes.
    :param min_tau: the minimum value of tau.
    :param initial_tau: the initial value of tau.
    :param tau_factor: the decay rate of tau (between 0 and 1)
    :param initial_const: initial value of the constant c.
    :param max_const: the maximum value of the constant c.
    :param const_factor: the rate of growth of the constant c.
    :param reduce_const: if True, the inital value of c is halved every
        time tau is reduced.
    :param warm_start: if True, use the previous adversarials as starting point
        for the next iteration.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm.
    :param max_iterations: the maximum number of iterations.
    :param abort_early: if set to true, abort early if getting stuck in local
        min.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """
    def __init__(self, predict, num_classes, min_tau=1/256,
                 initial_tau = 1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None):
        """Carlini Wagner LInfinity Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(CarliniWagnerLinfAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.predict = predict
        self.num_classes = num_classes
        self.min_tau = min_tau
        self.initial_tau = initial_tau
        self.tau_factor = tau_factor
        self.initial_const = initial_const
        self.max_const = max_const
        self.const_factor = const_factor
        self.reduce_const = reduce_const
        self.warm_start = warm_start
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _outputs_and_loss(self, x, modifiers, starting_atanh, y, const, taus):
        adversarials = tanh_rescale(starting_atanh + modifiers, self.clip_min, self.clip_max)

        outputs = self.predict(adversarials)
        y_onehot = to_one_hot(y, self.num_classes).float()

        real = (y_onehot * outputs).sum(dim=1)

        other = ((1.0 - y_onehot) * outputs - (y_onehot * TARGET_MULT)
                 ).max(dim=1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)

        loss1 = const * loss1

        image_dimensions = tuple(range(1, len(x.shape)))
        taus_shape = (-1,) + (1,) * (len(x.shape) - 1)

        penalties = torch.clamp(torch.abs(x - adversarials) - taus.view(taus_shape), min=0)
        loss2 = torch.sum(penalties, dim=image_dimensions)

        assert loss1.shape == loss2.shape

        loss = loss1 + loss2
        return outputs.detach(), loss

    def _successful(self, outputs, y):
        adversarial_labels = torch.argmax(outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, y)
        else:
            return ~torch.eq(adversarial_labels, y)

    def _run_attack(self, x, y, initial_const, taus, prev_adversarials):
        assert len(x) == len(taus)
        batch_size = len(x)
        best_adversarials = x.clone().detach()

        if self.warm_start:
            starting_atanh = self._get_arctanh_x(prev_adversarials.clone())
        else:
            starting_atanh = self._get_arctanh_x(x.clone())

        modifiers = torch.nn.Parameter(torch.zeros_like(starting_atanh))
        
        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=boolean_type, device=x.device)
        optimizer = optim.Adam([modifiers], lr=self.learning_rate)

        const = initial_const

        while torch.any(active) and const < self.max_const:
            for _ in range(self.max_iterations):
                optimizer.zero_grad()
                outputs, loss = self._outputs_and_loss(x[active], modifiers[active], starting_atanh[active], y[active], const, taus[active])
                total_loss = torch.sum(loss)

                total_loss.backward()
                optimizer.step()

                adversarials = tanh_rescale(starting_atanh + modifiers, self.clip_min, self.clip_max).detach()
                best_adversarials[active] = adversarials[active]

                # If early abortion is enabled, drop successful samples with a small loss
                # (The current adversarials are saved regardless of whether they are dropped)
                if self.abort_early:
                    successful = self._successful(outputs, y[active])
                    small_loss = loss < 0.0001 * const

                    drop = successful & small_loss
                    active[active] = ~drop

                if not active.any():
                    break

            # Give more weight to the output loss
            const *= self.const_factor

        return best_adversarials


    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        final_adversarials = x.clone()
        
        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=boolean_type, device=x.device)

        initial_const = self.initial_const
        taus = torch.ones((batch_size,), device=x.device) * self.initial_tau

        # The previous adversarials. This is used to perform a "warm start"
        # during optimization
        prev_adversarials = x.clone()

        while torch.any(active):
            adversarials = self._run_attack(x[active], y[active], initial_const, taus[active], prev_adversarials[active].clone())

            # Store the adversarials for the next iteration, even if they failed
            prev_adversarials[active] = adversarials

            # Drop failed adversarials (without saving them)
            adversarial_outputs = self.predict(adversarials)
            successful = self._successful(adversarial_outputs, y[active])
            active[active] = successful

            # Save the remaining adversarials
            final_adversarials[active] = adversarials[successful]
            
            # If the Linf distance is lower than tau and the adversarial is successful, use it as the new tau
            linf_distances = torch.max(torch.abs(final_adversarials - x).view(batch_size, -1), dim=1)[0]
            linf_lower = linf_distances < taus
            taus[linf_lower] = linf_distances[linf_lower]

            taus *= self.tau_factor
            
            if self.reduce_const:
                initial_const /= 2

            # Drop samples with a low tau
            low_tau = taus[active] <= self.min_tau
            active[active] = ~low_tau

        return final_adversarials