# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from advertorch.functional import FloatToIntSqueezing

from .base import Processor


class BitSqueezing(Processor):
    """
    Bit Squeezing.

    :param bit_depth: bit depth.
    :param vmin: min value.
    :param vmax: max value.
    """

    def __init__(self, bit_depth, vmin=0., vmax=1.):
        super(BitSqueezing, self).__init__()

        self.bit_depth = bit_depth
        self.max_int = 2 ** self.bit_depth - 1
        self.vmin = vmin
        self.vmax = vmax


    def forward(self, x):
        return FloatToIntSqueezing.apply(
            x, self.max_int, self.vmin, self.vmax)


class BinaryFilter(BitSqueezing):
    """
    Binary Filter.

    :param vmin: min value.
    :param vmax: max value.
    """

    def __init__(self, vmin=0., vmax=1.):
        super(BinaryFilter, self).__init__(bit_depth=1, vmin=vmin, vmax=vmax)
