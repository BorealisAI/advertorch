# Copyright (c) 2019-present, Francesco Croce
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
from torch.autograd.gradcheck import zero_gradients
import time

from .fast_adaptive_boundary import FABAttack


class FABWithThreshold(FABAttack):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    plus bound on the norm of the perturbations
    https://arxiv.org/abs/1907.02044
    
    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           upper bound on the norm of the perturbations
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            verbose=False,
            seed=0):

        super(FABWithThreshold, self).__init__(
            predict=predict, norm=norm, n_restarts=n_restarts,
            n_iter=n_iter, eps=eps, alpha_max=alpha_max, eta=eta, beta=beta,
            verbose=verbose)

        self.seed = seed

    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def attack_single_run(self, x, y, use_rand_start=False):
        startt = time.time()
        if len(x.shape) == self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        im2 = x.clone()
        la2 = y.clone()
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])

        if use_rand_start:
            if self.norm == 'Linf':
                t = 2 * torch.rand(x1.shape).to(self.device) - 1
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                            ) * t / (t.reshape([t.shape[0], -1]).abs()
                                     .max(dim=1, keepdim=True)[0]
                                     .reshape([-1, *[1]*self.ndims])) * .5
            elif self.norm == 'L2':
                t = torch.randn(x1.shape).to(self.device)
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                            ) * t / ((t ** 2)
                                     .view(t.shape[0], -1)
                                     .sum(dim=-1)
                                     .sqrt()
                                     .view(t.shape[0], *[1]*self.ndims)) * .5
            elif self.norm == 'L1':
                t = torch.randn(x1.shape).to(self.device)
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                            ) * t / (t.abs().view(t.shape[0], -1)
                                     .sum(dim=-1)
                                     .view(t.shape[0], *[1]*self.ndims)) / 2

            x1 = x1.clamp(0.0, 1.0)

        counter_iter = 0
        while counter_iter < self.n_iter:
            with torch.no_grad():
                df, dg = self.get_diff_logits_grads_batch(x1, la2)
                if self.norm == 'Linf':
                    dist1 = df.abs() / (1e-12 +
                                        dg.abs()
                                        .view(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1))
                elif self.norm == 'L2':
                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                        .view(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1).sqrt())
                elif self.norm == 'L1':
                    dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                        [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                else:
                    raise ValueError('norm not supported')
                ind = dist1.min(dim=1)[1]
                dg2 = dg[u1, ind]
                b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                                     .sum(dim=-1))
                w = dg2.reshape([bs, -1])

                if self.norm == 'Linf':
                    d3 = self.projection_linf(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                elif self.norm == 'L2':
                    d3 = self.projection_l2(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                elif self.norm == 'L1':
                    d3 = self.projection_l1(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)
                if self.norm == 'Linf':
                    a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                        .view(-1, *[1]*self.ndims)
                elif self.norm == 'L2':
                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                        .view(-1, *[1]*self.ndims)
                elif self.norm == 'L1':
                    a0 = d3.abs().sum(dim=1, keepdim=True)\
                        .view(-1, *[1]*self.ndims)
                a0 = torch.max(a0, 1e-8 * torch.ones(
                    a0.shape).to(self.device))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                alpha = torch.min(torch.max(a1 / (a1 + a2),
                                            torch.zeros(a1.shape)
                                            .to(self.device)),
                                  self.alpha_max * torch.ones(a1.shape)
                                  .to(self.device))
                x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                      (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                is_adv = self._get_predicted_label(x1) != la2

                if is_adv.sum() > 0:
                    ind_adv = is_adv.nonzero().squeeze()
                    ind_adv = self.check_shape(ind_adv)
                    if self.norm == 'Linf':
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                    elif self.norm == 'L2':
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                            .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                    elif self.norm == 'L1':
                        t = (x1[ind_adv] - im2[ind_adv])\
                            .abs().view(ind_adv.shape[0], -1).sum(dim=-1)
                    adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                        float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                        * (t >= res2[ind_adv]).float().reshape(
                        [-1, *[1]*self.ndims])
                    res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                        + res2[ind_adv] * (t >= res2[ind_adv]).float()
                    x1[ind_adv] = im2[ind_adv] + (
                        x1[ind_adv] - im2[ind_adv]) * self.beta

                counter_iter += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), bs) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[ind_succ] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y=None):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if y is None:
            y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        with torch.no_grad():
            acc = self.predict(x).max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    adv_curr = self.attack_single_run(
                        x_to_fool, y_to_fool, use_rand_start=(counter > 0))

                    output_curr = self.predict(adv_curr)
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                    if self.norm == 'Linf':
                        res = (x_to_fool - adv_curr).abs().view(
                            x_to_fool.shape[0], -1).max(1)[0]
                    elif self.norm == 'L2':
                        res = ((x_to_fool - adv_curr) ** 2).view(
                            x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                    elif self.norm == 'L1':
                        res = (x_to_fool - adv_curr).abs().view(
                            x_to_fool.shape[0], -1).sum(-1)
                    acc_curr = torch.max(acc_curr, res > self.eps)

                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[
                        ind_curr].clone()

                    if self.verbose:
                        print('restart {}'.format(counter),
                            '- target_class {}'.format(target_cl),
                            '- robust accuracy: {:.2%}'.format(
                                acc.float().mean()),
                            'at eps = {:.5f}'.format(self.eps),
                            '- cum. time: {:.1f} s'.format(
                                time.time() - startt))

        return adv

class FABTargeted(FABWithThreshold):
    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            verbose=False,
            seed=0,
            n_target_classes=9):
        """
        FAB with considering only one possible alternative class
        """
        super(FABTargeted, self).__init__(
            predict=predict, norm=norm, n_restarts=n_restarts,
            n_iter=n_iter, eps=eps, alpha_max=alpha_max, eta=eta, beta=beta,
            verbose=verbose, seed=seed)

        self.y_target = None
        self.n_target_classes = n_target_classes

    def get_diff_logits_grads_batch(self, imgs, la):
        la_target = self.y_target
        u = torch.arange(imgs.shape[0])

        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg

    def perturb(self, x, y=None):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if y is None:
            y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        with torch.no_grad():
            output = self.predict(x)
            la_sorted = output.sort(1)[1]
            acc = output.max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for target_cl in range(2, self.n_target_classes + 2):
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()

                        self.y_target = la_sorted[
                            ind_to_fool, -target_cl].clone()

                        adv_curr = self.attack_single_run(x_to_fool,
                            y_to_fool, use_rand_start=(counter > 0))

                        output_curr = self.predict(adv_curr)
                        acc_curr = output_curr.max(1)[1] == y_to_fool
                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs().view(
                                x_to_fool.shape[0], -1).max(1)[0]
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).view(
                                x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                        elif self.norm == 'L1':
                            res = (x_to_fool - adv_curr).abs().view(
                                x_to_fool.shape[0], -1).sum(-1)
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[
                            ind_curr].clone()

                        if self.verbose:
                            print('restart {}'.format(counter),
                                '- target_class {}'.format(target_cl),
                                '- robust accuracy: {:.2%}'.format(
                                    acc.float().mean()),
                                'at eps = {:.5f}'.format(self.eps),
                                '- cum. time: {:.1f} s'.format(
                                    time.time() - startt))

        return adv

