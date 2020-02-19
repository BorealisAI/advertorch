# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings

import numpy as np
import torch
import torchvision.transforms.functional as F

from advertorch.utils import torch_allclose
from advertorch.utils import clamp
from advertorch.utils import CIFAR10_MEAN
from advertorch.utils import CIFAR10_STD
from advertorch.utils import MNIST_MEAN
from advertorch.utils import MNIST_STD
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.utils import PerImageStandardize
from advertorch.utils import torch_flip
from advertorch_examples.utils import bchw2bhwc
from advertorch_examples.utils import bhwc2bchw


def test_mnist_normalize():
    # MNIST
    tensor = torch.rand((16, 1, 28, 28))
    normalize = NormalizeByChannelMeanStd(MNIST_MEAN, MNIST_STD)

    assert torch_allclose(
        torch.stack([F.normalize(t, MNIST_MEAN, MNIST_STD)
                     for t in tensor.clone()]),
        normalize(tensor))


def test_cifar10_normalize():
    # CIFAR10
    tensor = torch.rand((16, 3, 32, 32))
    normalize = NormalizeByChannelMeanStd(CIFAR10_MEAN, CIFAR10_STD)

    assert torch_allclose(
        torch.stack([F.normalize(t, CIFAR10_MEAN, CIFAR10_STD)
                     for t in tensor.clone()]),
        normalize(tensor))


def test_grad_through_normalize():
    tensor = torch.rand((2, 1, 28, 28))
    tensor.requires_grad_()
    mean = torch.tensor((0.,))
    std = torch.tensor((1.,))
    normalize = NormalizeByChannelMeanStd(mean, std)

    loss = (normalize(tensor) ** 2).sum()
    loss.backward()

    assert torch_allclose(2 * tensor, tensor.grad)


def _run_tf_per_image_standardization(imgs):
    import tensorflow as tf
    import tensorflow.image  # noqa: F401

    imgs = bchw2bhwc(imgs)
    placeholder = tf.placeholder(tf.float32, shape=imgs.shape)
    var_scaled = tf.map_fn(
        lambda img: tf.image.per_image_standardization(img), placeholder)

    with tf.Session() as sess:
        tf_scaled = sess.run(var_scaled, feed_dict={placeholder: imgs})
    return bhwc2bchw(tf_scaled)


def test_per_image_standardization():
    imgs = np.random.normal(
        scale=1. / (3072 ** 0.5), size=(10, 3, 32, 32)).astype(np.float32)
    per_image_standardize = PerImageStandardize()
    pt_scaled = per_image_standardize(torch.tensor(imgs)).numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf_scaled = _run_tf_per_image_standardization(imgs)
    assert np.abs(pt_scaled - tf_scaled).max() < 0.001


def test_clamp():

    def _convert_to_float(x):
        return float(x) if x is not None else None

    def _convert_to_batch_tensor(x, data):
        return x * torch.ones_like(data) if x is not None else None

    def _convert_to_single_tensor(x, data):
        return x * torch.ones_like(data[0]) if x is not None else None


    for min, max in [(-1, None), (None, 1), (-1, 1)]:

        data = 3 * torch.randn((11, 12, 13))
        case1 = clamp(data, min, max)
        case2 = clamp(data, _convert_to_float(min), _convert_to_float(max))
        case3 = clamp(data, _convert_to_batch_tensor(min, data),
                      _convert_to_batch_tensor(max, data))
        case4 = clamp(data, _convert_to_single_tensor(min, data),
                      _convert_to_single_tensor(max, data))

        assert torch.all(case1 == case2)
        assert torch.all(case2 == case3)
        assert torch.all(case3 == case4)


def test_flip():
    x = torch.randn(4, 5, 6, 7)
    assert (torch_flip(x, dims=(1, 2)) == torch.flip(x, dims=(1, 2))).all()
