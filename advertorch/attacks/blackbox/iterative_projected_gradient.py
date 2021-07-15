from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from tqdm import tqdm

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*torch.exp(lr*g)
    neg = (1-real_x)*torch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def gd_prior_step(x, g, lr):
    return x + lr*g

def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

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
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

def _check_param(param, x, name):
    if isinstance(param, float):
        new_param = param * torch.ones_like(x)
    elif isinstance(param, (np.ndarray, list)):
        new_param = torch.FloatTensor(param).to(x.device)  # type: ignore
    elif isinstance(param, torch.Tensor):
        new_param = param.to(x.device)  # type: ignore
    else:
        raise ValueError("Unknown format for {}".format(name))

    return new_param

#https://github.com/MadryLab/blackbox-bandits/blob/master/src/main.py
class BanditAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, 
            fd_eta, exploration, online_lr, order,
            loss_fn=None, 
            nb_iter=40,
            eps_iter=0.01,
            clip_min=0., clip_max=1.,
            targeted : bool = False
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
        x_adv = x.clone()

        eps = _check_param(self.eps, x.new_full((x.shape[0], )), 1, 'eps')
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

        proj_step = self.proj_maker(x, self.eps)
        
        #if so, we could just use the same estimate grad
        ndim = np.prod(list(x.shape[1:]))
        #The idea of this is that the gradient becomes more accurate as we
        #call it multiple times.
        #This gradient is learnt in an online fashion.

        prior = torch.zeros_like(x)

        for t in range(self.nb_iter):
            #before: # [nbatch, ndim, nsamples]
            #now: # [nbatch, ndim]
            exp_noise = exploration * torch.randn_like(prior)/(ndim**0.5)
            
            # Query deltas for finite difference estimator
            ##...this step needs to change
            q1 = F.normalize(prior + exp_noise, dim=-1)
            q2 = F.normalize(prior - exp_noise, dim=-1)
            # Loss points for finite difference estimator
            L1 = L(x_adv + self.fd_eta * q1) # L(prior + c*noise)
            L2 = L(x_adv + self.fd_eta * q2) # L(prior - c*noise)

            delta_L = (f1 - f2)/(self.fd_eta * self.exploration) #[nbatch]
            
            grad_est = delta_L * exp_noise

            prior = self.prior_step(prior, grad_est, self.online_lr)
            #upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1))
            x_adv = self.data_step(x_adv, prior, self.eps_iter)
            x_adv = proj_step(x_adv)
        
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return prior
        


