import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

from .utils import _check_param, _flatten

from advertorch.utils import to_one_hot

def sample_clamp(x, clip_min, clip_max):
    new_x = torch.maximum(x, clip_min[:, None, :])
    new_x = torch.minimum(new_x, clip_max[:, None, :])
    return new_x

def cw_log_loss(output, target, targeted=False, buff=1e-5):
    """
    :param outputs: pre-softmax/logits.
    :param target: true labels.
    :return: CW loss value.
    """
    num_classes = output.size(1)
    label_mask = to_one_hot(target, num_classes=num_classes).float()
    correct_logit = torch.log(torch.sum(label_mask * output, dim=1) + buff)
    wrong_logit = torch.log(torch.max((1. - label_mask) * output, dim=1)[0] + buff)

    if targeted:
        loss = -0.5 * F.relu(wrong_logit - correct_logit + 50.)
    else:
        loss = -0.5 * F.relu(correct_logit - wrong_logit + 50.)
    return loss


def select_best_example(x_orig, x_adv, order, losses, success):
    '''
    Given a collection of potential adversarial examples, select the best,
    according to either minimum input distortion or maximum output
    distortion.

    Also, return if adversarial example found (to avoid continuing the search)
    '''

    #x_orig: [B, F]
    #x_adv: [B, N, F]
    #success: [B, N], 0/1 if sample succeded
    #losses: [B, N]
    #dists: [B, N]

    if order == 'linf':
        dists = abs(x_orig[:, None, :] - x_adv).max(-1).values
    elif order == 'l2':
        dists = torch.sqrt( ((x_orig[:, None, :] - x_adv) ** 2).sum(-1) )
    else:
        raise ValueError('unsupported metric {}'.format(order))

    #[B, 1]
    #TODO: use the success mask!
    best_loss_ind = losses.argmin(-1)[:, None, None]
    best_loss_ind = best_loss_ind.expand(-1, -1, x_adv.shape[-1])
    #best_dist_ind = dists.argmin(-1)

    #return x_adv[:, 0, :]
    best_adv = torch.gather(x_adv, dim=1, index=best_loss_ind )
    return best_adv.squeeze(1)
    #return x_adv[:, best_loss_ind, :]

def n_attack(
        predict_fn, loss_fn, x, y, order, eps, clip_min, clip_max,
        mu_init=None, nb_samples=100, nb_iter=40, eps_iter=0.02, 
        sigma=0.1, targeted=False
    ):
    #TODO: check correspondence
    n_batch, n_dim = x.shape
    y_repeat = y.repeat(nb_samples, 1).T.flatten()

    #[B,F]
    if mu_init is None:
        mu_t = torch.FloatTensor(n_batch, n_dim).normal_() * 0.001
        mu_t = mu_t.to(x.device)
    else:
        mu_t = mu_init.clone()

    for _ in range(nb_iter):
        #Sample from N(0,I)
        #[B, N, F]
        gauss_samples = torch.FloatTensor(n_batch, nb_samples, n_dim).normal_()
        gauss_samples = gauss_samples.to(x.device)
        #print('samples nan?', torch.isnan(gauss_samples).any())

        #Compute gi = g(mu_t + sigma * samples)
        #[B, N, F]
        mu_samples = mu_t[:, None, :] + sigma * gauss_samples
        #print('mu_samples nan?', torch.isnan(mu_samples).any())

        #linf proj
        #[B, N, F]
        #TODO: change order...? l2 project?
        delta = sample_clamp(mu_samples, -eps[:, None], eps[:, None])
        #print('delta nan?', torch.isnan(delta).any())
        adv = x[:, None, :] + delta
        #print('adv nan?', torch.isnan(adv).any())
        adv = sample_clamp(adv, clip_min, clip_max)
        #print('clamp nan?', torch.isnan(adv).any())
        
        #shouldn't this go earlier?
        #[B * N, F]
        adv = adv.reshape(-1, n_dim)
        #[B * N, C]
        outputs = predict_fn(adv)
        #print('outputs?', torch.isnan(adv).any())
        #TODO: check that nothing is jumbled up

        #[B * N]
        if targeted:
            success_mask = torch.argmax(outputs, dim=-1) == y_repeat
        else:
            success_mask = torch.argmax(outputs, dim=-1) != y_repeat

        #[B, N]
        success_mask = success_mask.reshape(n_batch, nb_samples)
        #[B, N, C]
        adv = adv.reshape(n_batch, nb_samples, -1)
        #outputs = outputs.reshape(n_batch, self.nb_samples, -1)
        #

        #[B * N]
        losses = loss_fn(outputs, y_repeat, targeted=targeted)
        #print('losses nan?', torch.isnan(losses).any())
        #[B, N]
        losses = losses.reshape(n_batch, nb_samples)
        
        #[N]
        #z_score = (losses - np.mean(losses)) / (np.std(losses) + 1e-7)
        #[B, N]
        z_score = (losses - losses.mean(1)[:, None]) / (losses.std(1)[:, None] + 1e-7)
        #print('z_score nan?', torch.isnan(z_score).any())

        #mu_t: [B, F]
        #gauss_samples : [B,N,F]
        #z_score: [B,N]
        mu_t = mu_t + (eps_iter/(nb_samples*sigma)) * (z_score[:, :, None] * gauss_samples).sum(1)
        #print('mu_t nan?', torch.isnan(mu_t).any())

        #print('-' * 40)

        #TODO: should losses be increasing or decreasing?

    
    return adv, mu_t, losses, success_mask


class NAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, order='linf',
            loss_fn=None, 
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted : bool = False
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

        #Reference:
        #https://github.com/Cold-Winter/Nattack/blob/master/therm-adv/re_li_attack_notanh.py

    def perturb(self, x, y, mu_init=None):
        #[B, F]
        x, y = self._verify_and_process_inputs(x, y)
        shape, flat_x = _flatten(x)
        data_shape = tuple(shape[1:])
        n_batch, n_dim = flat_x.shape

        #[B]
        eps = _check_param(self.eps, x.new_full((x.shape[0],), 1), 'eps')
        #[B, F]
        clip_min = _check_param(self.clip_min, flat_x, 'clip_min')
        clip_max = _check_param(self.clip_max, flat_x, 'clip_max')

        def f(x):
            new_shape = (x.shape[0],) + data_shape
            input = x.reshape(new_shape)
            return self.predict(input)

        adv, _, losses, success_mask = n_attack(
            predict_fn=f, loss_fn=self.loss_fn, x=flat_x, y=y, order=self.order, 
            eps=eps, clip_min=clip_min, clip_max=clip_max,
            mu_init=None, nb_samples=self.nb_samples, nb_iter=self.nb_iter, 
            eps_iter=self.eps_iter, sigma=self.sigma, targeted=self.targeted
        )

        #return flat_x, adv, self.order, losses, success_mask

        adv = select_best_example(flat_x, adv, self.order, losses, success_mask)
        adv = adv.reshape(shape)

        return adv