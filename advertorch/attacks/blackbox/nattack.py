import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

from .utils import _check_param

from advertorch.utils import to_one_hot

# nb_samples = 300     # population size
# sigma = 0.1    # noise standard deviation
# alpha = 0.02  # learning rate
# # alpha = 0.001  # learning rate
# boxmin = 0
# boxmax = 1
# shift = (boxmin + boxmax) / 2. # 1/2 ... rescale based on clip_min, clip_max ... rename to shift
# scale = (boxmax - boxmin) / 2. # 1/2 ... rename to scale

#scale = (clip_max - clip_min) / 2
#shift = (clip_max + clip_min) / 2

#epsi = 0.031
#epsilon = 1e-30 #numerical safety factor (buffer)

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5

def encode_normal(z):
    return scale * np.tanh(z) + shift

def decode_input(x):
    return torch_arctanh((x - shift) / scale)

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + torch.clamp(new_x - orig, -eps, eps)
    return proj


def sample_clamp(x, clip_min, clip_max):
    new_x = torch.maximum(x, clip_min[:, None, :])
    new_x = torch.minimum(new_x, clip_max[:, None, :])
    return new_x


def cw_log_loss(output, target, targeted=False, buff=1e-30):
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

#TODO: Make classes for Linf, L2, ...
#TODO: accept custom losses

def n_attack(
        predict_fn, loss_fn, x, y, order, proj_step, eps, clip_min, clip_max,
        delta_init=None, nb_samples=100, nb_iter=40, eps_iter=0.02, 
        sigma=0.1, targeted=False
    ):
    #TODO: check correspondence
    n_batch, n_dim = x.shape
    y_repeat = y.repeat(nb_samples, 1).T.flatten()

    #[B,F]
    mu_t = torch.FloatTensor(n_batch, n_dim).normal_() * 0.001
    mu_t = mu_t.to(x.device)

    for _ in range(nb_iter):
        #Sample from N(0,I)
        #[B, N, F]
        gauss_samples = torch.FloatTensor(n_batch, nb_samples, n_dim).normal_()
        gauss_samples = gauss_samples.to(x.device)

        #Compute gi = g(mu_t + sigma * samples)
        #[B, N, F]
        mu_samples = mu_t[:, None, :] + sigma * gauss_samples

        #linf proj
        #[B, N, F]
        delta = sample_clamp(mu_samples, -eps[:, None], eps[:, None])
        adv = x[:, None, :] + delta
        adv = sample_clamp(adv, clip_min, clip_max)
        
        #shouldn't this go earlier?
        #[B * N, F]
        adv = adv.reshape(-1, n_dim)
        #[B * N, C]
        outputs = predict_fn(adv)
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
        #[B, N]
        losses = losses.reshape(n_batch, nb_samples)
        
        #[N]
        #z_score = (losses - np.mean(losses)) / (np.std(losses) + 1e-7)
        #[B, N]
        z_score = (losses - losses.mean(1)[:, None]) / (losses.std(1)[:, None] + 1e-7)

        #mu_t: [B, F]
        #gauss_samples : [B,N,F]
        #z_score: [B,N]
        mu_t = mu_t + (eps_iter/(nb_samples*sigma)) * (z_score[:, :, None] * gauss_samples).sum(1)

        #TODO: should losses be increasing or decreasing?

    
    return adv, losses, success_mask


class NAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, order='linf',
            loss_fn=None, 
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted : bool = False,
            query_limit = None
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
        
        self.proj_maker = l2_proj if order == 'l2' else linf_proj

        #If aiming for query efficiency, stop as soon as one adversarial
        #example is found.  Set to False to continue iteration and find best
        #adversarial example
        self.query_limit = query_limit


        #Reference:
        #https://github.com/Cold-Winter/Nattack/blob/master/therm-adv/re_li_attack_notanh.py
        #TODO: tanh?

    def perturb(self, x, y, delta_init=None):
        #[B, F]
        x, y = self._verify_and_process_inputs(x, y)

        n_batch, n_dim = x.shape

        #[B]
        eps = _check_param(self.eps, x.new_full((x.shape[0],), 1), 'eps')
        #[B, F]
        clip_min = _check_param(self.clip_min, x, 'clip_min')
        clip_max = _check_param(self.clip_max, x, 'clip_max')

        proj_step = self.proj_maker(x, eps)

        adv, losses, success_mask = n_attack(
            predict_fn=self.predict, loss_fn=self.loss_fn, x=x, y=y, order=self.order, 
            proj_step=proj_step, eps=eps, clip_min=clip_min, clip_max=clip_max,
            delta_init=None, nb_samples=self.nb_samples, nb_iter=self.nb_iter, 
            eps_iter=self.eps_iter, sigma=self.sigma, targeted=self.targeted
        )

        adv = select_best_example(x, adv, self.order, losses, success_mask)

        return adv

        