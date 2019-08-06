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
from advertorch.utils import CIFAR10_MEAN
from advertorch.utils import CIFAR10_STD
from advertorch.utils import MNIST_MEAN
from advertorch.utils import MNIST_STD
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.utils import PerImageStandardize
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
