# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import numpy as np
import torch

from advertorch.attacks import SinglePixelAttack
from advertorch.attacks import LocalSearchAttack
from advertorch.utils import predict_from_logits
from advertorch.test_utils import merge2dicts
from advertorch.test_utils import MLP

from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch_examples.utils import get_mnist_test_loader

import foolbox
from foolbox.attacks.localsearch import SinglePixelAttack as SPAfb
from foolbox.attacks.localsearch import LocalSearchAttack as LSAfb


NUM_CLASS = 10
BATCH_SIZE = 10
# TODO: need to make sure these precisions are enough
ATOL = 1e-4
RTOL = 1e-4

loader_test = get_mnist_test_loader(BATCH_SIZE)

data_iter = iter(loader_test)
img_batch, label_batch = data_iter.next()

# Setup the test MLP model
model = MLP()
model.eval()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, 'mlp.pkl'),
               map_location='cpu'))
model.to("cpu")

# foolbox single pixel attack do not succeed on this model
#   therefore using mlp.pkl
# from advertorch.test_utils import LeNet5
# model = LeNet5()
# model.eval()
# model.load_state_dict(
#     torch.load(os.path.join(TRAINED_MODEL_PATH,
#                             'mnist_lenet5_advtrained.pt')))
# model.to("cpu")




attack_kwargs = {
    SinglePixelAttack: {
        "fb_class": SPAfb,
        "kwargs": dict(
            max_pixels=50,
        ),
        "at_kwargs": dict(
            clip_min=0.0,
            clip_max=1.0,
            comply_with_foolbox=True,
        ),
        "fb_kwargs": dict(
            unpack=True,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    LocalSearchAttack: {
        "fb_class": LSAfb,
        "kwargs": dict(
            p=1.,
            r=1.5,
            d=10,
            t=100,
        ),
        "at_kwargs": dict(
            clip_min=0.0,
            clip_max=1.0,
            k=1,
            round_ub=100,
            comply_with_foolbox=True,
        ),
        "fb_kwargs": dict(
            R=100,
            unpack=True,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
}



def compare_at_fb(ptb_at, ptb_fb, atol, rtol):
    assert np.allclose(ptb_at, ptb_fb, atol=atol, rtol=rtol), \
        (np.abs(ptb_at - ptb_fb).max())


def compare_attacks(key, item):
    AdvertorchAttack = key
    fmodel = foolbox.models.PyTorchModel(
        model, bounds=(0, 1),
        num_classes=NUM_CLASS,
        cuda=False,
    )
    fb_adversary = item["fb_class"](fmodel)
    fb_kwargs = merge2dicts(item["kwargs"], item["fb_kwargs"])
    at_kwargs = merge2dicts(item["kwargs"], item["at_kwargs"])
    thresholds = item["thresholds"]
    at_adversary = AdvertorchAttack(model, **at_kwargs)
    x_at = at_adversary.perturb(img_batch, label_batch)
    y_logits = model(img_batch)
    y_at_logits = model(x_at)
    y_pred = predict_from_logits(y_logits)
    y_at_pred = predict_from_logits(y_at_logits)

    fb_successed_once = False
    for i, (x_i, y_i) in enumerate(zip(img_batch, label_batch)):
        # rule out when classification is wrong or attack is
        # unsuccessful (we test if foolbox attacks fails here)
        if y_i != y_pred[i:i + 1][0]:
            continue
        if y_i == y_at_pred[i:i + 1][0]:
            continue
        np.random.seed(233333)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_fb = fb_adversary(
                x_i.cpu().numpy(), label=int(y_i), **fb_kwargs)
        if x_fb is not None:
            compare_at_fb(x_at[i].cpu().numpy(), x_fb, **thresholds)
            fb_successed_once = True

    if not fb_successed_once:
        raise RuntimeError(
            "Foolbox never succeed, change your testing parameters!!!")


def test_single_pixel():
    compare_attacks(
        SinglePixelAttack,
        attack_kwargs[SinglePixelAttack],
    )


def test_local_search():
    compare_attacks(
        LocalSearchAttack,
        attack_kwargs[LocalSearchAttack],
    )


if __name__ == '__main__':
    pass
