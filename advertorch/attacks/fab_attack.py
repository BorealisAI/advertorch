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

from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin

EPS_DICT = {'Linf': .3, 'L2': 1., 'L1': 5.0}


class FABAttack(Attack, LabelMixin):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044

    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    :param device:        device to use ('cuda' or 'cpu')
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=-1,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            device='none',
            loss_fn='none'):
        """ FAB-attack implementation in pytorch """

        super(FABAttack, self).__init__(
            predict, loss_fn=None, clip_min=0., clip_max=1.)

        self.norm = norm
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps if eps > 0 else EPS_DICT[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.targeted = False
        self.device = device

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)
        g2 = self.compute_jacobian(im, y).detach()
        y2 = self.predict(imgs).detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def compute_jacobian(self, inputs, output):
        """
        from github.com/ast0414/adversarial-example/blob/master/craft.py
        """

        assert inputs.requires_grad
        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size()).to(self.device)
        grad_output = torch.zeros(*output.size()).to(self.device)
        if inputs.is_cuda:
            grad_output = grad_output.to(self.device)
            jacobian = jacobian.to(self.device)

        for i in range(num_classes):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)

            jacobian[i] = inputs.grad.data

        return torch.transpose(jacobian, dim0=0, dim1=1)

    def projection_linf(self, points_to_project, w_hyperplane, b_hyperplane):
        t = points_to_project.clone()
        w = w_hyperplane.clone()
        b = b_hyperplane.clone()

        ind2 = ((w * t).sum(1) - b < 0).nonzero().squeeze()
        ind2 = self.check_shape(ind2)
        w[ind2] *= -1
        b[ind2] *= -1

        c5 = (w < 0).float()
        a = torch.ones(t.shape).to(self.device)
        d = (a * c5 - t) * (w != 0).float()
        a -= a * (1 - c5)

        p = torch.ones(t.shape).to(self.device) * c5 - t * (2 * c5 - 1)
        indp = torch.argsort(p, dim=1)

        b = b - (w * t).sum(1)
        b0 = (w * d).sum(1)
        b1 = b0.clone()

        counter = 0
        indp2 = indp.unsqueeze(-1).flip(dims=(1, 2)).squeeze()
        u = torch.arange(0, w.shape[0])
        ws = w[u.unsqueeze(1), indp2]
        bs2 = - ws * d[u.unsqueeze(1), indp2]

        s = torch.cumsum(ws.abs(), dim=1)
        sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

        c = b - b1 > 0
        b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
        c_l = (b - b2 > 0).nonzero().squeeze()
        c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
        c_l = self.check_shape(c_l)
        c2 = self.check_shape(c2)

        lb = torch.zeros(c2.shape[0])
        ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
        nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
        counter2 = torch.zeros(lb.shape).long()

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long()
            indcurr = indp[c2, -counter2 - 1]
            b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
            c = b[c2] - b2 > 0
            ind3 = c.nonzero().squeeze()
            ind32 = (~c).nonzero().squeeze()
            ind3 = self.check_shape(ind3)
            ind32 = self.check_shape(ind32)
            lb[ind3] = counter4[ind3]
            ub[ind32] = counter4[ind32]
            counter += 1

        lb = lb.long()
        counter2 = 0

        if c_l.nelement != 0:
            lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]),
                                  torch.zeros(sb[c_l, -1].shape)
                                  .to(self.device))).unsqueeze(-1)
            d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

        lmbd_opt = (torch.max((b[c2] - sb[c2, lb]) / (-s[c2, lb]),
                              torch.zeros(sb[c2, lb].shape)
                              .to(self.device))).unsqueeze(-1)
        d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2]\
            + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])

        return d * (w != 0).float()

    def projection_l2(self, points_to_project, w_hyperplane, b_hyperplane):
        t = points_to_project.clone()
        w = w_hyperplane.clone()
        b = b_hyperplane.clone()

        c = (w * t).sum(1) - b
        ind2 = (c < 0).nonzero().squeeze()
        ind2 = self.check_shape(ind2)
        w[ind2] *= -1
        c[ind2] *= -1

        u = torch.arange(0, w.shape[0]).unsqueeze(1)

        r = torch.max(t / w, (t - 1) / w)
        u2 = torch.ones(r.shape).to(self.device)
        r = torch.min(r, 1e12 * u2)
        r = torch.max(r, -1e12 * u2)
        r[w.abs() < 1e-8] = 1e12
        r[r == -1e12] = -r[r == -1e12]
        rs, indr = torch.sort(r, dim=1)
        rs2 = torch.cat((rs[:, 1:],
                        torch.zeros(rs.shape[0], 1).to(self.device)), 1)
        rs[rs == 1e12] = 0
        rs2[rs2 == 1e12] = 0

        w3 = w ** 2
        w3s = w3[u, indr]
        w5 = w3s.sum(dim=1, keepdim=True)
        ws = w5 - torch.cumsum(w3s, dim=1)
        d = -(r * w).clone()
        d = d * (w.abs() > 1e-8).float()
        s = torch.cat(((-w5.squeeze() * rs[:, 0]).unsqueeze(1),
                      torch.cumsum((-rs2 + rs) * ws, dim=1) -
                      w5 * rs[:, 0].unsqueeze(-1)), 1)

        c4 = (s[:, 0] + c < 0)
        c3 = ((d * w).sum(dim=1) + c > 0)
        c6 = c4.nonzero().squeeze()
        c2 = ((1 - c4) * (1 - c3)).nonzero().squeeze()
        c6 = self.check_shape(c6)
        c2 = self.check_shape(c2)

        counter = 0
        lb = torch.zeros(c2.shape[0])
        ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
        nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
        counter2 = torch.zeros(lb.shape).long()

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long()
            c3 = s[c2, counter2] + c[c2] > 0
            ind3 = c3.nonzero().squeeze()
            ind32 = (~c3).nonzero().squeeze()
            ind3 = self.check_shape(ind3)
            ind32 = self.check_shape(ind32)
            lb[ind3] = counter4[ind3]
            ub[ind32] = counter4[ind32]
            counter += 1

        lb = lb.long()
        alpha = torch.zeros([1])

        if c6.nelement() != 0:
            alpha = c[c6] / w5[c6].squeeze(-1)
            d[c6] = -alpha.unsqueeze(-1) * w[c6]

        if c2.nelement() != 0:
            alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
            if torch.sum(ws[c2, lb] == 0) > 0:
                ind = (ws[c2, lb] == 0).nonzero().squeeze().long()
                ind = self.check_shape(ind)
                alpha[ind] = 0
            c5 = (alpha.unsqueeze(-1) > r[c2]).float()
            d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

        return d * (w.abs() > 1e-8).float()

    def projection_l1(self, points_to_project, w_hyperplane, b_hyperplane):
        t = points_to_project.clone()
        w = w_hyperplane.clone()
        b = b_hyperplane.clone()

        c = (w * t).sum(1) - b
        ind2 = (c < 0).nonzero().squeeze()
        ind2 = self.check_shape(ind2)
        w[ind2] *= -1
        c[ind2] *= -1

        r = torch.max(1 / w, -1 / w)
        r = torch.min(r, 1e12 * torch.ones(r.shape).to(self.device))
        rs, indr = torch.sort(r, dim=1)
        _, indr_rev = torch.sort(indr)

        u = torch.arange(0, w.shape[0]).unsqueeze(1)
        u2 = torch.arange(0, w.shape[1]).repeat(w.shape[0], 1)
        c6 = (w < 0).float()
        d = (-t + c6) * (w != 0).float()
        d2 = torch.min(-w * t, w * (1 - t))
        ds = d2[u, indr]
        ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
        s = torch.cumsum(ds2, dim=1)

        c4 = s[:, -1] < 0
        c2 = c4.nonzero().squeeze(-1)
        c2 = self.check_shape(c2)

        counter = 0
        lb = torch.zeros(c2.shape[0])
        ub = torch.ones(c2.shape[0]) * (s.shape[1])
        nitermax = torch.ceil(torch.log2(torch.tensor(s.shape[1]).float()))
        counter2 = torch.zeros(lb.shape).long()

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long()
            c3 = s[c2, counter2] > 0
            ind3 = c3.nonzero().squeeze()
            ind32 = (~c3).nonzero().squeeze()
            ind3 = self.check_shape(ind3)
            ind32 = self.check_shape(ind32)
            lb[ind3] = counter4[ind3]
            ub[ind32] = counter4[ind32]
            counter += 1

        lb2 = lb.long()

        if c2.nelement() != 0:
            alpha = -s[c2, lb2] / w[c2, indr[c2, lb2]]
            c5 = u2[c2].float() < lb.unsqueeze(-1).float()
            u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
            d[c2] = d[c2] * u3.float().to(self.device)
            d[c2, indr[c2, lb2]] = alpha

        return d * (w.abs() > 1e-8).float()

    def perturb(self, x, y=None):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        if self.device == 'none':
            self.device = x.device

        x = x.detach().clone().float().to(self.device)
        assert next(self.predict.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = replicate_input(x[pred])
        la2 = replicate_input(y[pred])
        if len(im2.shape) == 3:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < self.n_restarts:
            if counter_restarts > 0:
                if self.norm == 'Linf':
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, 1, 1, 1])
                                ) * t / (t.reshape([t.shape[0], -1]).abs()
                                         .max(dim=1, keepdim=True)[0]
                                         .reshape([-1, 1, 1, 1])) * .5
                elif self.norm == 'L2':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, 1, 1, 1])
                                ) * t / ((t ** 2)
                                         .sum(dim=(1, 2, 3), keepdim=True)
                                         .sqrt()) * .5
                elif self.norm == 'L1':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, 1, 1, 1])
                                ) * t / (t.abs()
                                         .sum(dim=(1, 2, 3), keepdim=True)) / 2

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 + dg.abs().sum(dim=(2, 3, 4)))
                    elif self.norm == 'L2':
                        dist1 = df.abs() / (1e-12 +
                                            (dg ** 2).sum(dim=(2, 3, 4)).sqrt())
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                    else:
                        raise ValueError('norm not supported')
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).sum(dim=(1, 2, 3)))
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
                            .unsqueeze(-1).unsqueeze(-1)
                    elif self.norm == 'L2':
                        a0 = (d3**2).sum(dim=1, keepdim=True).sqrt()\
                            .unsqueeze(-1).unsqueeze(-1)
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(dim=1, keepdim=True)\
                            .unsqueeze(-1).unsqueeze(-1)
                    a0 = torch.max(a0, 1e-8 * torch.ones(a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device))[0],
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
                                .sum(dim=(-3, -2, -1)).sqrt()
                        elif self.norm == 'L1':
                            t = (x1[ind_adv] - im2[ind_adv])\
                                .abs().sum(dim=(-3, -2, -1))
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).float()\
                            .reshape([-1, 1, 1, 1]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape([-1, 1, 1, 1])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (x1[ind_adv] -
                                                      im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        print('success rate: {:.0f}/{:.0f}'
              .format(ind_succ.float().sum(), corr_classified) +
              ' (on correctly classified points) in {:.1f} s'
              .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c
