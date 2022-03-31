# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from math import inf
from typing import Optional

import torch
import torch.nn.functional as F

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

from .utils import _check_param, _flatten, _make_projector

from advertorch.utils import to_one_hot


def cw_log_loss(output, target, targeted=False, buff=1e-5):
    """
    :param outputs: pre-softmax/logits.
    :param target: true labels.
    :return: CW loss value.
    """
    num_classes = output.size(1)
    label_mask = to_one_hot(target, num_classes=num_classes).float()
    correct_logit = torch.log(torch.sum(label_mask * output, dim=1) + buff)
    wrong_logit = torch.log(
        torch.max((1. - label_mask) * output, dim=1)[0] + buff)

    if targeted:
        loss = -0.5 * F.relu(wrong_logit - correct_logit + 50.)
    else:
        loss = -0.5 * F.relu(correct_logit - wrong_logit + 50.)
    return loss


def select_best_example(x_adv, losses):
    '''
    Given a collection of potential adversarial examples, select the best.

    :param x_adv: Candidate adversarial examples
        - shape [nbatch, nsample, ndim]
    :param losses: Loss values for each candidate exampe
        - shape [nbatch, nsample]
    '''
    best_loss_ind = losses.argmin(-1)[:, None, None]
    best_loss_ind = best_loss_ind.expand(-1, -1, x_adv.shape[-1])

    best_adv = torch.gather(x_adv, dim=1, index=best_loss_ind)
    return best_adv.squeeze(1)


def n_attack(
    predict_fn, loss_fn, x, y, projector,
    mu_init=None, nb_samples=100, nb_iter=40, eps_iter=0.02,
    sigma=0.1, targeted=False
):
    """
    Models the distribution of adversarial examples near an input data point.
    Similar to an evolutionary algorithm, but parameteric.

    Used as part of NAttack.

    :param predict: forward pass function.
    :param loss_fn: loss function
        - must accept tensors of shape [nbatch, pop_size, ndim]
    :param x: input tensor.
    :param y: label tensor.
        - if None and self.targeted=False, compute y as predicted
        labels.
        - if self.targeted=True, then y must be the targeted labels.
    :param projector: function to project the perturbation into the eps-ball
        - must accept tensors of shape [nbatch, pop_size, ndim]
    :param nb_samples: number of samples for  (default 100)
    :param nb_iter: number of iterations (default 40)
    :param eps_iter: attack step size (default 0.02).
    :param sigma: variance to control sample generation (default 0.1)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    n_batch, n_dim = x.shape
    y_repeat = y.repeat(nb_samples, 1).T.flatten()

    # [B,F]
    if mu_init is None:
        mu_t = torch.FloatTensor(n_batch, n_dim).normal_() * 0.001
        mu_t = mu_t.to(x.device)
    else:
        mu_t = mu_init.clone()

    # factor used to scale updates to mu_t
    alpha = eps_iter / (nb_samples * sigma)

    for _ in range(nb_iter):
        # Sample from N(0,I), shape [B, N, F]
        gauss_samples = torch.FloatTensor(n_batch, nb_samples, n_dim).normal_()
        gauss_samples = gauss_samples.to(x.device)

        # Compute gi = g(mu_t + sigma * samples), shape [B, N, F]
        mu_samples = mu_t[:, None, :] + sigma * gauss_samples
        delta = projector(mu_samples)
        adv = (x[:, None, :] + delta).reshape(-1, n_dim)
        outputs = predict_fn(adv)
        losses = loss_fn(outputs, y_repeat, targeted=targeted)
        losses = losses.reshape(n_batch, nb_samples)

        # Convert losses into z_scores
        z_score = (losses - losses.mean(1)
                   [:, None]) / (losses.std(1)[:, None] + 1e-7)

        # Update mu_t based on the z_scores
        mu_t = mu_t + alpha * (z_score[:, :, None] * gauss_samples).sum(1)

    adv = adv.reshape(n_batch, nb_samples, -1)

    return adv, mu_t, losses


class NAttack(Attack, LabelMixin):
    """
    Implements NAttack: https://arxiv.org/abs/1905.00441

    Disclaimers: Note that NAttack assumes the model outputs
    normalized probabilities.  Moreover, computations are broadcasted,
    so it is advisable to use smaller batch sizes when nb_samples is
    large.

    Hyperparams: sigma controls the variance for the generation of
    perturbations.

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param order: the order of maximum distortion (inf or 2)
    :param loss_fn: loss function (default None, NAttack uses CW loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param eps_iter: attack step size (default 0.02)
    :param sigma: variance to control sample generation (default 0.1)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
            self, predict, eps: float, order,
            loss_fn=None,
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted: bool = False
    ):

        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        super().__init__(predict, cw_log_loss, clip_min, clip_max)
        self.eps = eps
        self.order = order
        self.nb_samples = nb_samples
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.sigma = sigma
        self.targeted = targeted

    def perturb(
        self,
        x: torch.FloatTensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        # [B, F]
        x, y = self._verify_and_process_inputs(x, y)
        shape, flat_x = _flatten(x)
        data_shape = tuple(shape[1:])
        n_batch, n_dim = flat_x.shape

        # [B]
        eps = _check_param(self.eps, x.new_full((x.shape[0],), 1), 'eps')
        # [B, F]
        clip_min = _check_param(self.clip_min, flat_x, 'clip_min')
        clip_max = _check_param(self.clip_max, flat_x, 'clip_max')

        def f(x):
            new_shape = (x.shape[0],) + data_shape
            input = x.reshape(new_shape)
            return self.predict(input)


        projector = _make_projector(
            eps, self.order, flat_x, clip_min, clip_max
        )

        adv, _, losses = n_attack(
            predict_fn=f, loss_fn=self.loss_fn, x=flat_x, y=y,
            projector=projector, nb_samples=self.nb_samples,
            nb_iter=self.nb_iter, eps_iter=self.eps_iter, sigma=self.sigma,
            targeted=self.targeted
        )

        adv = select_best_example(adv, losses)
        adv = adv.reshape(shape)

        return adv


class LinfNAttack(NAttack):
    """
    NAttack with order=inf

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param order: the order of maximum distortion (inf or 2)
    :param loss_fn: loss function (default None, NAttack uses CW loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param eps_iter: attack step size (default 0.02)
    :param sigma: variance to control sample generation (default 0.1)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
            self, predict, eps: float,
            loss_fn=None,
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted: bool = False
    ):

        super(LinfNAttack, self).__init__(
            predict=predict, eps=eps, order=inf, loss_fn=loss_fn,
            nb_samples=nb_samples, nb_iter=nb_iter, eps_iter=eps_iter,
            sigma=sigma, clip_min=clip_min, clip_max=clip_max,
            targeted=targeted
        )



class L2NAttack(NAttack):
    """
    NAttack with order=2

    :param predict: forward pass function.
    :param eps: maximum distortion.
    :param order: the order of maximum distortion (inf or 2)
    :param loss_fn: loss function (default None, NAttack uses CW loss)
    :param nb_samples: population size (default 100)
    :param nb_iter: number of iterations (default 40)
    :param eps_iter: attack step size (default 0.02)
    :param sigma: variance to control sample generation (default 0.1)
    :param clip_min: mininum value per input dimension (default 0.)
    :param clip_max: mininum value per input dimension (default 1.)
    :param targeted: if the attack is targeted (default False)
    """

    def __init__(
            self, predict, eps: float,
            loss_fn=None,
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted: bool = False
    ):

        super(L2NAttack, self).__init__(
            predict=predict, eps=eps, order=2, loss_fn=loss_fn,
            nb_samples=nb_samples, nb_iter=nb_iter, eps_iter=eps_iter,
            sigma=sigma, clip_min=clip_min, clip_max=clip_max,
            targeted=targeted
        )
