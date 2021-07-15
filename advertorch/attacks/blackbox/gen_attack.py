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

#TODO: rename this import, it is wrong.
#TODO: difference between linf_eps and clip_min/clip_max
#TODO: remove tqdm and all print statements
from advertorch.utils import clamp as batch_clamp
from advertorch.utils import replicate_input

class GeneticL2Attack(Attack, LabelMixin):
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

    def __init__(
            self, predict, N: int, eps: float, loss_fn=None, 
            clip_min=0., clip_max=1.,
            alpha_min: float = 0.15,
            rho_min: float = 0.1,
            G_max: int = 20,
            num_elites: int = 5,
            targeted : bool = False
            ):
        """
        Create an instance of the PGDAttack.

        """
        super().__init__(predict, loss_fn, clip_min, clip_max)

        self.N = N
        self.alpha_min = alpha_min
        self.rho_min = rho_min
        self.G_max = G_max
        self.num_elites = num_elites
        self.eps = eps
        self.targeted = targeted

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
        
        if isinstance(self.eps, float):
            eps = torch.FloatTensor([self.eps] * x.shape[0]).to(  # type: ignore
                x.device
            )
        elif isinstance(self.eps, (np.ndarray, list)):
            eps = torch.FloatTensor(self.eps).to(x.device)  # type: ignore
        elif isinstance(self.eps, torch.Tensor):
            eps = self.eps.to(x.device)  # type: ignore

        else:
            raise ValueError("Unknown linf_eps format.")

        if isinstance(self.clip_min, float):
            clip_min = self.clip_min * torch.ones_like(x)
        elif isinstance(self.clip_min, (np.ndarray, list)):
            clip_min = torch.FloatTensor(self.clip_min).to(x.device)  # type: ignore
        elif isinstance(self.clip_min, torch.Tensor):
            clip_min = self.clip_min.to(x.device)  # type: ignore

        else:
            raise ValueError("Unknown clip_min format.")

        if isinstance(self.clip_max, float):
            clip_max = self.clip_max * torch.ones_like(x)
        elif isinstance(self.clip_max, (np.ndarray, list)):
            clip_max = torch.FloatTensor(self.clip_max).to(x.device)  # type: ignore
        elif isinstance(self.clip_max, torch.Tensor):
            clip_max = self.clip_max.to(x.device)  # type: ignore

        else:
            raise ValueError("Unknown clip_max format.")

        #adv_ -> delta
        adv_ex = []
        success = []
        G = []
        for ind in tqdm(range(x.shape[0])):
            adv_, succ, g = self._perturb_sample(
                x[ind : ind + 1],  # noqa: E203
                y[ind],
                eps[ind].item(),
                clip_min[ind : ind + 1],  # noqa: E203
                clip_max[ind : ind + 1],  # noqa: E203
                self.N,
                self.alpha_min,
                self.rho_min,
                self.G_max,
                self.num_elites,
            )
            G.append(g)
            success.append(succ)
            #adv_ = self.decode(adv_ + x[ind : ind + 1])  # noqa: E203
            adv_ = (adv_ + x[ind : ind + 1])  # noqa: E203
            adv_ex.append(adv_)

        adv_ex = torch.FloatTensor(  # type: ignore
            [xi.cpu().numpy()[0] for xi in adv_ex]
        ).to(
            x.device
        )

        return adv_ex, success, G

    def _perturb_sample(
        self,
        X: torch.FloatTensor,
        Y: torch.Tensor, #should be optional
        linf_eps: float,
        bmin: torch.Tensor,
        bmax: torch.Tensor,
        N: int,
        alpha_min: float,
        rho_min: float,
        G_max: int,
        num_elites: int,
    ) -> Tuple[torch.Tensor, bool, int]:
        """
        Algo 1 from : https://arxiv.org/pdf/1805.11090.pdf
        Used as a helper function in `generate_counterexamples`.
        Runs the GenAttack for each sample. The input samples
        should be encoded (ga.encode).

        Args:
            X: original encoded example
            Y: true label
            linf_eps: maximum distance
            bmin: minimum bound for perturbed `x` (encoded).
            bmax: maximum bound for perturbed `x` (encoded).
            N : size of population.
            alpha_min: min mutation range.
            rho_min: min mutation probability.
            G_max: # of generations.
            num_elites: number of top members to keep for the next generation.

        Returns:
            torch.Tensor: the best adversarial example found by GenAttack.
            bool:  if the attack is successful.
            int: number of generations ran to find the adversarial example.
        """
        device = X.device

        # initialize population
        dims = list(X.size())
        dims[0] = N

        bmin_ = torch.cat(N * [bmin], dim=0)
        bmax_ = torch.cat(N * [bmax], dim=0)

        #this "population" corresponds to "delta"
        population = torch.empty(dims, device=device).uniform_(-linf_eps, linf_eps)
        population = batch_clamp(population + X, bmin_, bmax_) - X

        # initialize variables used in while loop
        count = 1  # Start with an initial population - so count starts as 1
        crit = 1e-5
        last_best = num_i = num_plat = 0
        adv_attack = False
        # Continue until max num. of iterations or get an adversarial example
        while not adv_attack and count < G_max:
            if count % 100 == 0:
                print("Generation " + str(count))
            # Find fitness for every individual and save the best fitness
            fitness = self.fitness(population + X, Y)
            best_fit = min(fitness)

            # Check to if fitness last two generations is the same,
            # update num_plat
            if abs(best_fit - last_best) <= crit:
                num_i += 1
                if num_i % 100 == 0 and num_i != 0:
                    print("Plateau at Generation " + str(count))
                    num_plat += 1
            else:
                num_i = 0

            # Get sorted indices (Ascending!)
            _, sorted_inds = fitness.sort()
            # Initialize new population by adding the elites
            new_pop = torch.zeros_like(population)
            for i in range(num_elites):
                new_pop[i] = population[sorted_inds[i]]

            # The best individual is the one with the best fitness
            best = new_pop[0]

            adv_attack = self.valid_attack(best + X, Y)

            # If not a true adversarial example need to go to next generation
            if not adv_attack:
                alpha = max(alpha_min, 0.5 * (0.9 ** num_plat))
                rho = max(rho_min, 0.4 * (0.9 ** num_plat))
                # Softmax fitnesse
                soft_fit = F.softmax(-fitness, dim=0)  # Negate fitness since we're trying to minimize
                # need to get apply selection and get a new population
                child_pop = self.selection(population, soft_fit, X, alpha, rho, linf_eps, num_elites, bmin, bmax)
                new_pop[num_elites:] = child_pop
                population = new_pop
                count += 1
                # Need to retain best fitness
                last_best = best_fit

        return best, adv_attack, count

    def mutation_op(
        self,
        cur_pop: torch.Tensor,
        x_orig: torch.Tensor,
        alpha: float,
        rho: float,
        delta_max: float,
        bmin: torch.Tensor,
        bmax: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perturbs the sample with random values to generate new population.

        Args:
            cur_pop: the current population
            x_orig :  the image we are using for the attack
            alpha: mutation range
            rho: mutation probability
            delta_max: maximum distance
            bmin: minimum bound for perturbed `x` (encoded).
            bmax: maximum bound for perturbed `x` (encoded).

        Returns:
            new population
        """
        step_noise = alpha * delta_max
        perturb_noise = torch.empty_like(cur_pop).uniform_(-step_noise, step_noise)
        mask = torch.empty_like(cur_pop).bernoulli_(rho)
        mutated_pop = perturb_noise * mask + cur_pop
        bmin_ = torch.cat(cur_pop.shape[0] * [bmin], dim=0)
        bmax_ = torch.cat(cur_pop.shape[0] * [bmax], dim=0)

        try:
            clamped_mutation_pop = batch_clamp(mutated_pop + x_orig, bmin_, bmax_) - x_orig
        except AssertionError:
            import ipdb; ipdb.set_trace()
        normalized_pop = torch.clamp(clamped_mutation_pop, -delta_max, delta_max)
        return normalized_pop

    def fitness(self, batch: torch.Tensor, target_class: torch.Tensor):
        """
        The objective function to be solved.

        This source code is inspired by:
        https://github.com/nesl/adversarial_genattack/blob/
        2304cdc2a49d2c16b9f43821ad8a29d664f334d1/genattack_tf2.py#L39

        Args:
            batch: a batch of examples
            target_class: the class label of x (as an integer)

        Returns:
            Tensor of fitness
        """
        with torch.no_grad():
            probs = self.predict(batch)
            log_probs = torch.log(probs)
            s = 1.0 - probs[:, target_class]
            f = log_probs[:, target_class] - torch.log(s)
            f = torch.clamp(f.flatten(), -1000, 1000)  # clamping to avoid the "all inf" problem

            if self.targeted:
                f = -f
            return f

    def selection(
        self, population, soft_fit, data, alpha, rho, delta_max, num_elite, bmin, bmax,
    ):
        """
        Runs the crossover and permutation to generate a new generation.

        Args:
            population: the population of individuals
            soft_fit: the softmax of the fitness
            data: the input value to find a perturbation for
            alpha: mutation range
            rho: mutation probability
            num_elite: the number of elites to carry on from the previous
                generations

        Returns:
            mut_child_pop: Returns the mutated population of children
        """

        # Crossover
        cdims = list(population.size())
        child_pop_size = population.size()[0] - num_elite
        cdims[0] = child_pop_size
        child_pop = torch.empty(cdims, device=data.device)
        # Roulette
        roulette = Categorical(probs=soft_fit)
        for i in range(child_pop_size):
            parent1_idx = roulette.sample()
            soft_fit_nop1 = soft_fit.clone() + 0.0001  # Incrementing by epsilon to avoid the "all zeros" problem
            soft_fit_nop1[parent1_idx] = 0
            roulette2 = Categorical(probs=soft_fit_nop1)
            parent2_idx = roulette2.sample()
            child = self.crossover(
                population[parent1_idx], population[parent2_idx], soft_fit[parent1_idx], soft_fit[parent2_idx],
            )
            child_pop[i] = child

        # Mutation
        mut_child_pop = self.mutation_op(child_pop, data, alpha, rho, delta_max, bmin, bmax)

        return mut_child_pop

    def crossover(
        self, parent1: torch.Tensor, parent2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Element-wise crossover

        Args:
            parent1: individual in old population
            parent2: individual in old population
            p1: softmaxed fitness for parent1
            p2: softmaxed fitness for parent2

        Returns:
            child: new individual from mating of parents
        """
        p = p1 / (p1 + p2)
        mask = torch.empty_like(parent1).bernoulli_(p)
        child = mask * parent1 + (1 - mask) * parent2  # type: ignore
        return child

    def valid_attack(self, data: torch.Tensor, t: torch.Tensor) -> bool:
        """
        Args:
            data: perturbation + original sample
            t: true class label
        Returns:
            adv_attack: Whether the new sample is an adversarial attack
        """
        #TODO: should change this based on targeted/untargeted
        adv_attack = False
        t_out = self.predict(data)
        t_pred = t_out.argmax(dim=1, keepdim=True)
        if t != t_pred:
            adv_attack = True

        return adv_attack
