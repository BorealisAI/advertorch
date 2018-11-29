# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from scipy import ndimage

from advertorch.test_utils import generate_data_model_on_img
from advertorch.utils import torch_allclose
from advertorch.defenses import BinaryFilter
from advertorch.defenses import MedianSmoothing2D

data, label, model = generate_data_model_on_img()


def test_binary_filter():
    assert torch_allclose(BinaryFilter()(data), data > 0.5)


def test_median_filter():
    # XXX: doesn't pass when kernel_size is even
    # XXX: when kernel_size is odd, pixels on the boundaries are different
    kernel_size = 3
    padding = kernel_size // 2
    rval_scipy = ndimage.filters.median_filter(
        data.detach().numpy(), size=(1, 1, kernel_size, kernel_size))
    rval = MedianSmoothing2D(kernel_size=kernel_size)(data).detach().numpy()
    assert np.allclose(rval_scipy[:, :, padding:-padding, padding:-padding],
                       rval[:, :, padding:-padding, padding:-padding])


# TODO: correctness test of GaussianSmoothing2D and AverageSmoothing2D

if __name__ == '__main__':
    test_binary_filter()
    test_median_filter()
