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


from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from tqdm import tqdm

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful, rand_init_delta

#from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from .utils import _check_param

#TODO: rename this import, it is wrong.
#TODO: difference between linf_eps and clip_min/clip_max
#TODO: remove tqdm and all print statements
from advertorch.utils import clamp as batch_clamp
from advertorch.utils import replicate_input

RHOMIN = 0.1
ALPHAMIN = 0.15

def gen_attack_score(output, target, targeted=False):
    n_class = output.shape[-1]
    y_onehot = F.one_hot(target, num_classes=n_class)
    y_onehot = 2 * y_onehot - 1 #(+1) only for the positive class

    score = (y_onehot[:, None, :] * output).sum(-1)

    #with this convention, we try to maximize fitness
    if not targeted:
        score = -score

    return score

def compute_fitness(predict_fn, loss_fn, adv_pop, y, targeted=False):
    """
    The objective function to be solved.

    This source code is inspired by:
    https://github.com/nesl/adversarial_genattack/blob/
    2304cdc2a49d2c16b9f43821ad8a29d664f334d1/genattack_tf2.py#L39

    Args:
        predict_fn: The blackbox model
        pop_t: candidate adversarial examples
        y: the class label of x (as an integer)

    Returns:
        Tensor of fitness (shape: n_batch)
    """
    #OBSERVE / TODO:
    #GenAttack does not require a transform_fn, since it operates on the 
    #probs directly

    #population shape: [B, N, F]
    n_batch, n_samples, n_dim = adv_pop.shape
    #reshape to [B * N, F]
    adv_pop = adv_pop.reshape(-1, n_dim)
    #output shape: [B * N, C]
    probs = predict_fn(adv_pop)
    
    #reshape to [B, N, C]
    probs = probs.reshape(n_batch, n_samples, -1)
    log_probs = torch.log(probs)
    #y shape: [B, N]
    #y_onehot shape: [B, N, C]

    #shape: [B, N]
    fitness = loss_fn(log_probs, y, targeted=targeted)

    #TODO: clamp to avoid infinites/numerical instabilities

    return fitness

def crossover(p1, p2, probs):
    u = torch.rand(*p1.shape)
    return torch.where(probs[:, :, None] > u, p1, p2)

def sample_clamp(x, clip_min, clip_max):
    new_x = torch.maximum(x, clip_min[:, None, :])
    new_x = torch.minimum(new_x, clip_max[:, None, :])
    return new_x

def selection(pop_t, fitness, tau):
    n_batch, nb_samples, n_dim = pop_t.shape

    probs = F.softmax(fitness / tau, dim=1)
    cum_probs = probs.cumsum(-1)
    #Edge case, u1 or u2 is greater than max(cum_probs)
    cum_probs[:, -1] = 1. + 1e-7

    #parents: instead of selecting one elite, select two, and generate
    #a new child to create a population around
    #do this multiple times, for each N

    #sample parent 1 from pop_t according to probs
    #sample parent 2 from pop_t according to probs

    #sample from multinomial
    u1, u2 = torch.rand(2, n_batch, nb_samples)
    
    #out of the original N samples, we draw another N samples
    #this requires us to compute the following broadcasted comparison
    p1ind = -((cum_probs[:, :, None] > u1[:, None, :]).long()).sum(1) + nb_samples
    p2ind = -((cum_probs[:, :, None] > u2[:, None, :]).long()).sum(1) + nb_samples

    parent1 = torch.gather(
        pop_t, dim=1, index=p1ind[:, :, None].expand(-1, -1, n_dim)
    )
    
    parent2 = torch.gather(
        pop_t, dim=1, index=p2ind[:, :, None].expand(-1, -1, n_dim)
    )

    fp1 = torch.gather(fitness, dim=1, index=p1ind)
    fp2 = torch.gather(fitness, dim=1, index=p2ind)
    crossover_prob = fp1 / (fp1 + fp2)

    return crossover(parent1, parent2, crossover_prob)

def mutation(pop_t, alpha, rho, eps):
    #alpha and eps both have shape [B]
    perturb_noise = (2 * torch.rand(*pop_t.shape) - 1)
    perturb_noise = perturb_noise * alpha[:, None, None] * eps[:, None, None]

    #TODO: consistent mulinomial sampling

    mask = (torch.rand(*pop_t.shape) > rho[:, None, None]).float()

    return pop_t + mask * perturb_noise

#TODO: account for other projections
def linf_project(pop_t, x, eps, clip_min, clip_max):    
    delta = sample_clamp(pop_t, -eps[:, None], eps[:, None])
    #mutated_pop = x[:, None, :] + delta
    #mutated_pop = sample_clamp(mutated_pop, clip_min, clip_max)
    delta = sample_clamp(x[:, None, :] + delta, clip_min, clip_max
        ) - x[:, None, :]

    return delta

class GenAttackScheduler():
    def __init__(self, x, alpha_init=0.4, rho_init=0.5, decay=0.9):
        n_batch = x.shape[0]

        self.n_batch = n_batch
        self.crit = 1e-5

        self.best_val = torch.zeros(n_batch).to(x.device)
        self.num_i = torch.zeros(n_batch).to(x.device)
        self.num_plateaus = torch.zeros(n_batch).to(x.device)

        self.rho_min = RHOMIN * torch.ones(n_batch).to(x.device)
        self.alpha_min = ALPHAMIN * torch.ones(n_batch).to(x.device)

        self.zeros = torch.zeros_like(self.num_i)

        #check alpha, rho
        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.decay = decay

        self.alpha = alpha_init * torch.ones(n_batch).to(x.device)
        self.rho = rho_init * torch.ones(n_batch).to(x.device)

    def update(self, elite_val):
        stalled = abs(elite_val - self.best_val) <= self.crit
        self.num_i = torch.where(stalled, self.num_i + 1, self.zeros)
        new_plateau = (self.num_i % 100 == 0) & (self.num_i != 0)
        self.num_plateaus = torch.where(
            new_plateau, self.num_plateaus + 1, self.num_plateaus
        )
        
        #update alpha and rho
        self.rho = torch.maximum(
            self.rho_min, self.rho_init * self.decay ** self.num_plateaus
        )
        self.alpha = torch.maximum(
            self.alpha_min, self.alpha_init * self.decay ** self.num_plateaus
        )

        self.best_val = torch.maximum(elite_val, self.best_val)


def gen_attack(predict_fn, loss_fn, x, y, eps, clip_min, clip_max, nb_samples, nb_iter, tau=0.1, 
        alpha_init=0.4, rho_init=0.5, decay=0.9, pop_init=None, scheduler=None, targeted=False
    ):
    #alpha: mutation range
    #rho: mutation probability
    #nb_samples: population size
    #nb_iter: number of generations
    #tau: sampling temperature
    #pop_init: initial population

    #alpha, rho, and tau all control exploration
    #they are all constants
    #rho is a mutation probability
    #alpha is a step size

    #TODO: temperature scaling?

    #TODO: numerical instabilities check

    n_batch, n_dim = x.shape
    #y_repeat = y.repeat(nb_samples, 1).T.flatten()

    #[B,F]
    if pop_init is None:
        #Sample from Uniform(-1, 1)
        #shape: [B, N, F]
        pop_t = 2 * torch.rand(n_batch, nb_samples, n_dim) - 1
        #Sample from Uniform(-eps, eps)
        pop_t = eps[:, None, None] * pop_t
        pop_t = pop_t.to(x.device)

        #pop_t = x[:, None, :] + pop_t
    else:
        pop_t = pop_init.clone()

    if scheduler is None:
        scheduler = GenAttackScheduler(x, alpha_init, rho_init, decay)

    #TODO: alpha_init, rho_init
    #per sample?

    inds = torch.arange(n_batch).to(x.device)

    for _ in range(nb_iter):
        adv = x[:, None, :] + pop_t
        #shape: [B, N]
        fitness = compute_fitness(predict_fn, loss_fn, adv, y, targeted=targeted)
        #shape: [1, B, 1]
        #TODO: argmax or argmin?
        #TODO: doesn't max return argmax?
        #elite_ind = fitness.argmax(-1) 
        #[B]
        #elite_val = fitness.max(-1).values
        elite_val, elite_ind = fitness.max(-1)
        #shape: [B, F]
        #TODO: test this works across devices
        #elite_adv = fitness[inds, elite_ind, :]
        elite_adv = adv[inds, elite_ind, :]
        #shape: [B]
        elite_pred = predict_fn(elite_adv).argmax(-1)

        #shape: [B, N]
        #select which members will move onto the next generation
        children = selection(pop_t, fitness, tau)

        #apply mutations and clipping
        #add mutated child to next generation (ie update pop_t)
        pop_t = mutation(children, scheduler.alpha, scheduler.rho, eps)
        #TODO: account for other projections
        pop_t = linf_project(pop_t, x, eps, clip_min, clip_max)

        #Update params based on plateaus
        scheduler.update(elite_val)

    #Partial attack:
    #If success, leave alone, and turn x -> x[~success]
    return elite_adv, pop_t, scheduler

class GeneticLinfAttack(Attack, LabelMixin):
    """
    Runs GenAttack https://arxiv.org/abs/1805.11090.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.

    Args:
        X: original samples [#samples, #feature].
        Y: true labels
        eps: maximum epsilon size. If float the same
            epsilon will be used for all samples. Different
            epsilons can be provided using an array.
        N: size of population
        alpha_min: min mutation range.
        rho_min: min mutation probability.
        G_max: # of generations
        num_elites: number of top members to keep for the next generation

    Returns:
        torch.Tensor: generated adversarial examples.
        list: if the adversarial attack is successful
        list: number of generations used for each sample
    """
    #gen_attack(predict_fn, x, y, eps, clip_min, clip_max, nb_samples, nb_iter, tau=0.1, 
    #    alpha_init=0.4, rho_init=0.5, pop_init=None, scheduler=None, targeted=False
    #)
    def __init__(
            self, predict, eps: float, order='linf',
            loss_fn=None, 
            nb_samples=100,
            nb_iter=40,
            tau=0.1,
            alpha_init=0.4,
            rho_init=0.5,
            decay=0.9,
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

        super().__init__(predict, gen_attack_score, clip_min, clip_max)

        self.eps = eps
        self.order = order
        self.nb_samples = nb_samples
        self.nb_iter = nb_iter
        self.targeted = targeted

        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.decay = decay
        self.tau = tau

        #If aiming for query efficiency, stop as soon as one adversarial
        #example is found.  Set to False to continue iteration and find best
        #adversarial example
        self.query_limit = query_limit

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
        #[B, F]
        clip_min = _check_param(self.clip_min, x, 'clip_min')
        clip_max = _check_param(self.clip_max, x, 'clip_max')

        elite_adv, pop_t, scheduler = gen_attack(
            predict_fn=self.predict, loss_fn=self.loss_fn, x=x, y=y, 
            eps=eps, clip_min=clip_min, clip_max=clip_max, 
            nb_samples=self.nb_samples, nb_iter=self.nb_iter, tau=self.tau,
            alpha_init=self.alpha_init, rho_init=self.rho_init, 
            decay=self.decay, pop_init=None, scheduler=None, 
            targeted=self.targeted
        )

        return elite_adv