# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

from .base import Processor


class MedianSmoothing2D(Processor):
    """
    Median Smoothing 2D.

    :param kernel_size: aperture linear size; must be odd and greater than 1.
    :param stride: stride of the convolution.
    """

    def __init__(self, kernel_size=3, stride=1):
        super(MedianSmoothing2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        padding = int(kernel_size) // 2
        if _is_even(kernel_size):
            # both ways of padding should be fine here
            # self.padding = (padding, 0, padding, 0)
            self.padding = (0, padding, 0, padding)
        else:
            self.padding = _quadruple(padding)


    def forward(self, x):
        x = F.pad(x, pad=self.padding, mode="reflect")
        x = x.unfold(2, self.kernel_size, self.stride)
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(x.shape[:4] + (-1, )).median(dim=-1)[0]
        return x


class ConvSmoothing2D(Processor):
    """
    Conv Smoothing 2D.

    :param kernel_size: size of the convolving kernel.
    """

    def __init__(self, kernel):
        super(ConvSmoothing2D, self).__init__()
        self.filter = _generate_conv2d_from_smoothing_kernel(kernel)

    def forward(self, x):
        return self.filter(x)


class GaussianSmoothing2D(ConvSmoothing2D):
    """
    Gaussian Smoothing 2D.

    :param sigma: sigma of the Gaussian.
    :param channels: number of channels in the output.
    :param kernel_size: aperture size.
    """

    def __init__(self, sigma, channels, kernel_size=None):
        kernel = _generate_gaussian_kernel(sigma, channels, kernel_size)
        super(GaussianSmoothing2D, self).__init__(kernel)


class AverageSmoothing2D(ConvSmoothing2D):
    """
    Average Smoothing 2D.

    :param channels: number of channels in the output.
    :param kernel_size: aperture size.
    """

    def __init__(self, channels, kernel_size):
        kernel = torch.ones((channels, 1, kernel_size, kernel_size)) / (
            kernel_size * kernel_size)
        super(AverageSmoothing2D, self).__init__(kernel)


def _generate_conv2d_from_smoothing_kernel(kernel):
    channels = kernel.shape[0]
    kernel_size = kernel.shape[-1]

    if _is_even(kernel_size):
        raise NotImplementedError(
            "Even number kernel size not supported yet, kernel_size={}".format(
                kernel_size))

    filter_ = nn.Conv2d(
        in_channels=channels, out_channels=channels, kernel_size=kernel_size,
        groups=channels, padding=kernel_size // 2, bias=False)

    filter_.weight.data = kernel
    filter_.weight.requires_grad = False
    return filter_


def _generate_gaussian_kernel(sigma, channels, kernel_size=None):

    if kernel_size is None:
        kernel_size = _round_to_odd(2 * 2 * sigma)

    vecx = torch.arange(kernel_size).float()
    vecy = torch.arange(kernel_size).float()
    gridxy = _meshgrid(vecx, vecy)
    mean = (kernel_size - 1) / 2.
    var = sigma ** 2

    gaussian_kernel = (
        1. / (2. * math.pi * var) *
        torch.exp(-(gridxy - mean).pow(2).sum(dim=0) / (2 * var))
    )

    gaussian_kernel /= torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def _round_to_odd(f):
    return math.ceil(f) // 2 * 2 + 1


def _meshgrid(vecx, vecy):
    gridx = vecx.repeat(len(vecy), 1)
    gridy = vecy.repeat(len(vecx), 1).t()
    return torch.stack([gridx, gridy])


def _is_even(x):
    return int(x) % 2 == 0
