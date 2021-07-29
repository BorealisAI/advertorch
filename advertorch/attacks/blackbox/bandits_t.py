from typing import List, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from tqdm import tqdm

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.utils import clamp

from .utils import _check_param

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*torch.exp(lr*g)
    neg = (1-real_x)*torch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1


def norm(x):
    return torch.sqrt( (x ** 2).sum(-1))

def gd_prior_step(x, g, lr):
    return x + lr*g

def l2_data_step(x, g, lr):
    return x + lr*F.normalize(g, dim=-1)

def linf_step(x, g, lr):
    return x + lr*torch.sign(g)

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float().unsqueeze(-1)
        x = (orig + eps[:, None] * F.normalize(delta, dim=-1))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = torch.minimum(new_x - orig, eps[:, None])
        delta = torch.maximum(delta, -eps[:, None])
        return orig + delta
    return proj


def bandit_attack(
        x, y, loss_fn, prior_step, data_step, proj_step, clip_min, clip_max, 
        prior_init=None, fd_eta=0.01, exploration=0.01, online_lr=0.1, nb_iter=40,
        eps_iter=0.01,
    ):
    #if so, we could just use the same estimate grad
    ndim = np.prod(list(x.shape[1:]))
    #The idea of this is that the gradient becomes more accurate as we
    #call it multiple times.
    #This gradient is learnt in an online fashion.

    adv = x.clone()

    if prior_init is None:
        prior = torch.zeros_like(x)
    else:
        prior = prior_init.clone()

    for t in range(nb_iter):
        #before: # [nbatch, ndim, nsamples]
        #now: # [nbatch, ndim]
        exp_noise = exploration * torch.randn_like(prior)/(ndim**0.5)
        
        # Query deltas for finite difference estimator
        ##...this step needs to change
        q1 = F.normalize(prior + exp_noise, dim=-1)
        q2 = F.normalize(prior - exp_noise, dim=-1)
        # Loss points for finite difference estimator
        L1 = loss_fn(adv + fd_eta * q1) # L(prior + c*noise)
        L2 = loss_fn(adv + fd_eta * q2) # L(prior - c*noise)

        delta_L = (L1 - L2)/(fd_eta * exploration) #[nbatch]
        
        grad_est = delta_L * exp_noise

        prior = prior_step(prior, grad_est, online_lr)
        #upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1))
        adv = data_step(adv, prior, eps_iter)
        adv = proj_step(adv)
    
        #TODO: check this clamping is correct

        adv = torch.maximum(adv, clip_min)
        adv = torch.minimum(adv, clip_max)

    return adv, prior

#https://github.com/MadryLab/blackbox-bandits/blob/master/src/main.py
class BanditAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, order,
            fd_eta=0.01, exploration=0.01, online_lr=0.1,
            loss_fn=None, 
            nb_iter=40,
            eps_iter=0.01,
            clip_min=0., clip_max=1.,
            targeted : bool = False,
            query_limit=None
            ):

        super().__init__(predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.fd_eta = fd_eta
        self.exploration = exploration
        self.online_lr = online_lr
        self.prior_step = gd_prior_step if order == 'l2' else eg_step
        self.targeted = targeted

        self.nb_iter = nb_iter
        self.eps_iter = eps_iter

        self.data_step = l2_data_step if order == 'l2' else linf_step

        self.proj_maker = l2_proj if order == 'l2' else linf_proj

        self.query_limit = None

    def perturb(  # type: ignore
        self,
        x: torch.FloatTensor,
        y: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[bool], List[int]]:
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        eps = _check_param(self.eps, x.new_full((x.shape[0],), 1), 'eps')
        clip_min = _check_param(self.clip_min, x, 'clip_min')
        clip_max = _check_param(self.clip_max, x, 'clip_max')
        #sample using mean param
        #possibly, this would mean NESWrapper has a normal distribution
        #as a param
        
        #TODO: (1) adapt for multiple outputs
        #TODO: (2) test
        #TODO: (3) figure out better way of "storing" prior?
        # ... BANDIT IS NOT A TYPE OF GRADIENT ESTIMATOR
        # ... there is a dual optimization problem

        def L(x): #loss func
            output = self.predict(x)
            loss = self.loss_fn(output, y)
            return loss

        proj_step = self.proj_maker(x, eps)

        adv, prior = bandit_attack(
            x, y, loss_fn=L, prior_step=self.prior_step, data_step=self.data_step, 
            proj_step=proj_step, clip_min=clip_min, clip_max=clip_max, 
            prior_init=None,  fd_eta=self.fd_eta, exploration=self.exploration,
            online_lr=self.online_lr, nb_iter=self.nb_iter, eps_iter=self.eps_iter
        )
        
        return adv
        


