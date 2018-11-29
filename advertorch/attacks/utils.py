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

import numpy as np
import torch

from advertorch.utils import clamp
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import batch_multiply


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(0, 1)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    else:
        error = "Only ord = inf and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def is_successful(y1, y2, targeted):
    if targeted is True:
        return y1 == y2
    else:
        return y1 != y2


class AttackConfig(object):
    # a convenient class for generate an attack/adversary instance

    def __init__(self):
        self.kwargs = {}

        for mro in reversed(self.__class__.__mro__):
            if mro in (AttackConfig, object):
                continue
            for kwarg in mro.__dict__:
                if kwarg in self.AttackClass.__init__.__code__.co_varnames:
                    self.kwargs[kwarg] = mro.__dict__[kwarg]
                else:
                    # make sure we don't specify wrong kwargs
                    assert kwarg in ["__module__", "AttackClass", "__doc__"]

    def __call__(self, *args):
        adversary = self.AttackClass(*args, **self.kwargs)
        print(self.AttackClass, args, self.kwargs)
        return adversary
