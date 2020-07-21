# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product, repeat

import torch
import numpy as np
from .base import Attack
from .base import LabelMixin

_MESHGRIDS = {}


def make_meshgrid(x):
    bs, _, _, dim = x.shape
    device = x.get_device()
    key = (dim, bs, device)
    if key in _MESHGRIDS:
        return _MESHGRIDS[key]
    space = torch.linspace(-1, 1, dim)
    meshgrid = torch.meshgrid([space, space])
    gridder = torch.cat([meshgrid[1][..., None],
                         meshgrid[0][..., None]], dim=2)
    grid = gridder[None, ...].repeat(bs, 1, 1, 1)
    ones = torch.ones(grid.shape[:3] + (1,))
    final_grid = torch.cat([grid, ones], dim=3)
    expanded_grid = final_grid[..., None]
    _MESHGRIDS[key] = expanded_grid
    return expanded_grid


def unif(size, mini, maxi):
    args = {"from": mini, "to": maxi}
    return torch.FloatTensor(size=size).uniform_(**args)


def make_slice(a, b, c):
    to_cat = [a[None, ...], b[None, ...], c[None, ...]]
    return torch.cat(to_cat, dim=0)


def make_mats(rots, txs, w, h):
    # rots: degrees
    # txs: % of image dim
    rots = rots * 0.01745327778  # deg to rad
    txs = txs * 2
    cosses = torch.cos(rots)
    sins = torch.sin(rots)
    top_slice = make_slice(cosses, -sins, txs[:, 0])[None, ...].permute(
        [2, 0, 1])
    bot_slice = make_slice(sins, cosses, txs[:, 1])[None, ...].permute(
        [2, 0, 1])
    mats = torch.cat([top_slice, bot_slice], dim=1)
    mats = mats[:, None, None, :, :]
    mats = mats.repeat(1, w, h, 1, 1)
    return mats


def transform(x, rots, txs):
    assert x.shape[2] == x.shape[3]
    w = x.shape[2]
    h = x.shape[3]
    device = x.device
    
    with torch.no_grad():
        meshgrid = make_meshgrid(x).to(device)
        tfm_mats = make_mats(rots, txs, w, h).to(device)
        new_coords = torch.matmul(tfm_mats, meshgrid)
        new_coords = new_coords.squeeze_(-1)
        new_image = torch.nn.functional.grid_sample(x, new_coords,
                                                    align_corners=False)
        return new_image


class SpatialTransformAttack2(Attack, LabelMixin):
    """
    Spatially Transformed Attack (Engstrom et al. 2019)
    http://proceedings.mlr.press/v97/engstrom19a.html

    :param predict: forward pass function.
    :param spatial_constraint: max rot and max trans.
    :param num_rot: the number of rotation direction grid search
    :param num_trans: the number of translation direction grid search
    :param random_tries: the number of random search
    :param attack_type: attack search type random|grid
    :param loss_fn: loss function
    :param clip_min: minimum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted
    """
    
    def __init__(self, predict, spatial_constraint={'rot': 30, 'trans': 0.1},
                 num_rot=31, num_trans=5, random_tries=10,
                 attack_type='random', loss_fn=None,
                 clip_min=0.0, clip_max=1.0, targeted=False):
        super(SpatialTransformAttack2, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.predict = predict
        self.attack_type = attack_type
        self.targeted = targeted
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.spatial_constraint = [spatial_constraint['trans'],
                                   spatial_constraint['trans'],
                                   spatial_constraint['rot']]
        if self.attack_type == 'grid':
            self.granularity = [num_trans, num_trans, num_rot]
        elif self.attack_type == 'random':
            self.random_tries = random_tries
    
    def perturb(self, x_nat, y=None):
        x_nat, y = self._verify_and_process_inputs(x_nat, y)
        if self.attack_type == 'grid':
            return self.perturb_grid(x_nat, y, -1)
        elif self.attack_type == 'random':
            return self.perturb_grid(x_nat, y, self.random_tries)
        else:
            raise NotImplementedError()
    
    def perturb_grid(self, x_nat, y, random_tries=-1):
        device = x_nat.device
        n = len(x_nat)
        if random_tries > 0:
            # subsampling this list from the grid is a bad idea, instead we
            # will randomize each example from the full continuous range
            grid = [(42, 42, 42) for _ in range(random_tries)]  # dummy list
        else:  # exhaustive grid
            grid = product(*list(np.linspace(-ll, ll, num=g)
                                 for ll, g in zip(self.spatial_constraint,
                                                  self.granularity)))
        worst_x = x_nat.clone()
        worst_t = torch.zeros([n, 3]).to(device)
        max_xent = torch.zeros(n).to(device)
        all_correct = torch.ones(n, dtype=torch.bool).to(device)
        if hasattr(self.loss_fn, 'reduction'):
            self.org_reduction = self.loss_fn.reduction
            if self.loss_fn.reduction != 'none':
                self.loss_fn.reduction = 'none'
        else:
            print('loss_fn has been replaced by torch.nn.CrossEntropyLoss '
                  'because reduction none is not available. '
                  'If you want to use custom loss, '
                  'please implement reduction=none')
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        
        for tx, ty, r in grid:
            if random_tries > 0:
                # randomize each example separately
                t = np.stack([np.random.uniform(-ll, ll, n) for ll in
                              self.spatial_constraint], axis=1)
            else:
                t = np.stack(list(repeat([tx, ty, r], n)))
            x = x_nat
            t = torch.from_numpy(t).type(torch.float32).to(device)
            x = transform(x, t[:, 2], t[:, :2])
            with torch.no_grad():
                logits = self.predict(x)
            # get the index of the max log-probability
            pred = logits.detach().max(1)[1]
            cur_correct = pred.eq(y)
            if self.targeted:
                cur_xent = -self.loss_fn(logits, y)  # Reverse the sign
            else:
                cur_xent = self.loss_fn(logits, y)
            
            # Select indices to update: we choose the misclassified
            # transformation of maximum xent (or just highest xent
            # if everything else if correct).
            idx = (cur_xent > max_xent) & (cur_correct == all_correct)
            idx = idx | (cur_correct < all_correct)
            max_xent = torch.max(cur_xent, max_xent)
            all_correct = cur_correct & all_correct
            idx = idx.unsqueeze(-1)  # shape (bsize, 1)
            worst_t = torch.where(idx, t, worst_t)  # shape (bsize, 3)
            idx = idx.unsqueeze(-1)
            idx = idx.unsqueeze(-1)  # shape (bsize, 1, 1, 1)
            worst_x = torch.where(idx, x, worst_x, )  # shape (bsize, w, h, c)
        if hasattr(self, 'org_reduction'):
            self.loss_fn.reduction = self.org_reduction
        return worst_x
