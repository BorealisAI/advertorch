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

import os
import sys
import pathlib

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from advertorch.test_utils import LeNet5


ROOT_PATH = os.path.expanduser("~/.advertorch")
DATA_PATH = os.path.join(ROOT_PATH, "data")
MNIST_PATH = os.path.join(DATA_PATH, "mnist")
CIFAR10_PATH = os.path.join(DATA_PATH, "cifar10")

path_of_this_module = os.path.dirname(sys.modules[__name__].__file__)
TRAINED_MODEL_PATH = os.path.join(path_of_this_module, "trained_models")


def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def get_mnist_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_train"
    return loader


def get_mnist_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_test"
    return loader


def get_cifar10_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CIFAR10_PATH, train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_train"
    return loader


def get_cifar10_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CIFAR10_PATH, train=False, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_test"
    return loader


def get_mnist_lenet5_clntrained():
    filename = "mnist_lenet5_clntrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.eval()
    model.name = "MNIST LeNet5 standard training"
    # TODO: also described where can you find this model, and how is it trained
    return model


def get_mnist_lenet5_advtrained():
    filename = "mnist_lenet5_advtrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.eval()
    model.name = "MNIST LeNet 5 PGD training according to Madry et al. 2018"
    # TODO: also described where can you find this model, and how is it trained
    return model


def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


def _imshow(img):
    import matplotlib.pyplot as plt
    img = bchw2bhwc(img.detach().cpu().numpy())
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis("off")


class ImageNetClassNameLookup(object):

    def _load_list(self):
        import json
        with open(self.json_path) as f:
            class_idx = json.load(f)
        self.label2classname = [
            class_idx[str(k)][1] for k in range(len(class_idx))]

    def __init__(self):
        self.json_url = ("https://s3.amazonaws.com/deep-learning-models/"
                         "image-models/imagenet_class_index.json")
        self.json_path = os.path.join(DATA_PATH, "imagenet_class_index.json")
        if os.path.exists(self.json_path):
            self._load_list()
        else:
            import urllib
            urllib.request.urlretrieve(self.json_url, self.json_path)
            self._load_list()


    def __call__(self, label):
        return self.label2classname[label]


def get_panda_image():
    img_path = os.path.join(DATA_PATH, "panda.jpg")
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_panda_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_panda_image()
    else:
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_panda_image()


mkdir(ROOT_PATH)
mkdir(DATA_PATH)
mkdir(MNIST_PATH)
