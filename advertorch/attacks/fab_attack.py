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

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
#import numpy as np

from advertorch.utils import calc_l2distsq
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin
from .utils import is_successful

torch.set_default_tensor_type('torch.cuda.FloatTensor')

EPS_DICT = {'Linf': .3, 'L2': 1., 'L1': 5.0}


class FABattack(Attack, LabelMixin):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044

    :param predict:       forward pass function.
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(self,
                 predict,
                 norm='Linf',
                 n_restarts=1,
                 n_iter=100,
                 eps=-1,
                 alpha_max=0.1,
                 eta=1.05,
                 beta=0.9):
      """ FAB-attack implementation in pytorch """
      loss_fn = None

      super(FABattack, self).__init__(
            predict, loss_fn, clip_min=0., clip_max=1.)
            
      self.norm = norm
      self.n_restarts = n_restarts
      self.n_iter = n_iter
      self.eps = eps if eps > 0 else EPS_DICT[norm]
      self.alpha_max = alpha_max
      self.eta = eta
      self.beta = beta
      self.targeted = False
      
    def get_diff_logits_grads_batch(self, imgs, la):
      im = Variable(imgs.clone(), requires_grad=True)
      with torch.enable_grad(): y = self.predict(im)
      g2 = self.compute_jacobian(im, y).detach()
      y2 = self.predict(imgs).detach()
      df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
      dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
      df[torch.arange(imgs.shape[0]), la] = 1e10
      
      return df, dg
    
    def compute_jacobian(self, inputs, output):
      """ from https://github.com/ast0414/adversarial-example/blob/master/craft.py """
      
      assert inputs.requires_grad
      
      num_classes = output.size()[1]
      
      jacobian = torch.zeros(num_classes, *inputs.size())
      grad_output = torch.zeros(*output.size())
      if inputs.is_cuda:
      	grad_output = grad_output.cuda()
      	jacobian = jacobian.cuda()
      
      for i in range(num_classes):
      	zero_gradients(inputs)
      	grad_output.zero_()
      	grad_output[:, i] = 1
      	output.backward(grad_output, retain_graph=True)
      	jacobian[i] = inputs.grad.data
      
      return torch.transpose(jacobian, dim0=0, dim1=1)
      
    def projection_linf(self, points_to_project, w_hyperplane, b_hyperplane):
      t = points_to_project.clone().float()
      w = w_hyperplane.clone().float()
      b = b_hyperplane.clone().float()
      d = torch.zeros(t.shape).float()
      
      ind2 = ((w*t).sum(1) - b < 0).nonzero()
      w[ind2] *= -1
      b[ind2] *= -1
      
      c5 = (w < 0).type(torch.cuda.FloatTensor)
      a = torch.ones(t.shape).cuda()
      d = (a*c5 - t)*(w != 0).type(torch.cuda.FloatTensor)
      a -= a*(1 - c5)
      
      p = torch.ones(t.shape)*c5 - t*(2*c5 - 1)
      indp = torch.argsort(p, dim=1)
  
      b = b - (w*t).sum(1)
      b0 = (w*d).sum(1)
      b1 = b0.clone()
  
      counter = 0
      indp2 = indp.unsqueeze(-1).flip(dims=(1,2)).squeeze()
      u = torch.arange(0, w.shape[0])
      ws = w[u.unsqueeze(1), indp2]
      bs2 = - ws*d[u.unsqueeze(1), indp2]
      
      s = torch.cumsum(ws.abs(), dim=1)
      sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)
      
      c = b - b1 > 0
      b2 = sb[u, -1] - s[u, -1]*p[u, indp[u, 0]]
      c_l = (b - b2 > 0).nonzero().squeeze()
      c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
      
      lb = torch.zeros(c2.shape[0])
      ub = torch.ones(c2.shape[0])*(w.shape[1] - 1)
      nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
      counter2 = torch.zeros(lb.shape).type(torch.cuda.LongTensor)
      
      while counter < nitermax:
        counter4 = torch.floor((lb + ub)/2)
        counter2 = counter4.type(torch.cuda.LongTensor)
        indcurr = indp[c2, -counter2 - 1]
        b2 = sb[c2, counter2] - s[c2, counter2]*p[c2, indcurr]
        c = b[c2] - b2 > 0
        ind3 = c.nonzero().squeeze()
        ind32 = (~c).nonzero().squeeze()
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1
      
      lb = lb.cpu().numpy().astype(int)
      counter2 = 0
      
      if c_l.nelement != 0:  
        lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1])/(-s[c_l, -1]), torch.zeros(sb[c_l, -1].shape))).unsqueeze(-1)
        d[c_l] = (2*a[c_l] - 1)*lmbd_opt
        
      lmbd_opt = (torch.max((b[c2] - sb[c2, lb])/(-s[c2, lb]), torch.zeros(sb[c2, lb].shape))).unsqueeze(-1)
      d[c2] = torch.min(lmbd_opt, d[c2])*c5[c2] + torch.max(-lmbd_opt, d[c2])*(1-c5[c2])
  
      return (d*(w != 0).type(torch.cuda.FloatTensor))
      
    def perturb(self, x, y):
      x, y = x.detach().clone().float().cuda(), y.detach().clone().long().cuda()
      y_pred = self._get_predicted_label(x)
      pred = y_pred == y
      corr_classified = pred.float().sum()
      print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
      pred = pred.nonzero().squeeze()
      
      # runs the attack only on correctly classified points
      im2 = replicate_input(x[pred])
      la2 = replicate_input(y[pred])
      bs = im2.shape[0]
      u1 = torch.arange(bs)
      adv = im2.clone()
      adv_c = x.clone()
      res2 = 1e10*torch.ones([bs])
      res_c = torch.zeros([x.shape[0]])
      x1 = im2.clone()
      x0 = im2.clone().reshape([bs, -1])
      counter_restarts = 0
      
      while counter_restarts < self.n_restarts:
        if counter_restarts > 0:
          t = 2*torch.rand(x1.shape) - 1
          x1 = im2 + torch.min(res2, self.eps*torch.ones(res2.shape)).reshape([-1,1,1,1])*t/t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1,1,1,1])*0.5
          x1 = x1.clamp(0.0, 1.0)
        
        counter_iter = 0
        while counter_iter < self.n_iter:
          df, dg = self.get_diff_logits_grads_batch(x1, la2)
          dist1 = df.abs() / (1e-8 + dg.abs().sum(dim=(2,3,4)))
          ind = dist1.min(dim=1)[1]
          dg2 = dg[u1, ind]
          b = (- df[u1, ind] + (dg2*x1).sum(dim=(1,2,3)))
          w = dg2.reshape([bs, -1])
          
          d3 = self.projection_linf(torch.cat((x1.reshape([bs, -1]),x0),0), torch.cat((w, w), 0), torch.cat((b, b),0))
          d1 = torch.reshape(d3[:bs], x1.shape)
          d2 = torch.reshape(d3[-bs:], x1.shape)
          a0 = d3.abs().max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
          a0 = torch.max(a0, 1e-8*torch.ones(a0.shape))
          a1 = a0[:bs]
          a2 = a0[-bs:]
          alpha = torch.min(torch.max(a1/(a1 + a2), torch.zeros(a1.shape))[0], self.alpha_max*torch.ones(a1.shape))
          x1 = ((x1 + self.eta*d1)*(1 - alpha) + (im2 + d2*self.eta)*alpha).clamp(0.0, 1.0)
          
          is_adv = self._get_predicted_label(x1) != la2
          
          if is_adv.sum() > 0:
            ind_adv = is_adv.nonzero().squeeze()
            t = (x1[ind_adv] - im2[ind_adv]).reshape([ind_adv.shape[0], -1]).abs().max(dim=1)[0]
            adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).float().reshape([-1,1,1,1]) + adv[ind_adv]*(t >= res2[ind_adv]).float().reshape([-1,1,1,1])
            res2[ind_adv] = t * (t < res2[ind_adv]).float() + res2[ind_adv]*(t >= res2[ind_adv]).float()
            x1[ind_adv] = im2[ind_adv] + (x1[ind_adv] - im2[ind_adv])*self.beta
          
          counter_iter += 1
          
        counter_restarts += 1
      
      ind_succ = res2 < 1e10
      print('success rate: {:.0f}/{:.0f} (on correctly classified points)'.format(ind_succ.float().sum(), corr_classified))
      #print('check norms: ', (im2 - adv).abs().reshape([bs, -1]).max(dim=1)[0][:10])
      
      res_c[pred] = res2*ind_succ.float() + 1e10*(1 - ind_succ.float())
      ind_succ = ind_succ.nonzero().squeeze()
      #print(ind_succ, pred)
      adv_c[pred[ind_succ]] = adv[ind_succ].clone()
      #print('check norms: ', (x - adv_c).abs().reshape([x.shape[0], -1]).max(dim=1)[0][:10])
      
      return res2, adv_c