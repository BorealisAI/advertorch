# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import itertools

import torch
import torch.nn as nn

from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import GradientAttack
from advertorch.attacks import L2BasicIterativeAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L1PGDAttack
from advertorch.attacks import SparseL1DescentAttack
from advertorch.attacks import MomentumIterativeAttack
from advertorch.attacks import FastFeatureAttack
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import DDNL2Attack
from advertorch.attacks import ElasticNetL1Attack
from advertorch.attacks import LBFGSAttack
from advertorch.attacks import JacobianSaliencyMapAttack
from advertorch.attacks import SpatialTransformAttack
from advertorch.utils import CarliniWagnerLoss
from advertorch.utils import torch_allclose

from advertorch.test_utils import NUM_CLASS
from advertorch.test_utils import BATCH_SIZE
from advertorch.test_utils import batch_consistent_attacks
from advertorch.test_utils import general_input_attacks
from advertorch.test_utils import image_only_attacks
from advertorch.test_utils import label_attacks
from advertorch.test_utils import feature_attacks
from advertorch.test_utils import targeted_only_attacks

from advertorch.test_utils import vecdata
from advertorch.test_utils import veclabel
from advertorch.test_utils import vecmodel
from advertorch.test_utils import imgdata
from advertorch.test_utils import imglabel
from advertorch.test_utils import imgmodel


xent_loss = nn.CrossEntropyLoss(reduction="sum")
cw_loss = CarliniWagnerLoss()
mse_loss = nn.MSELoss(reduction="sum")
smoothl1_loss = nn.SmoothL1Loss(reduction="sum")

label_criteria = (xent_loss, cw_loss)
feature_criteria = (smoothl1_loss, mse_loss)

cuda = "cuda"
cpu = "cpu"

devices = (cpu, cuda) if torch.cuda.is_available() else (cpu, )

attack_kwargs = {
    GradientSignAttack: {},
    GradientAttack: {},
    LinfBasicIterativeAttack: {"nb_iter": 5},
    L2BasicIterativeAttack: {"nb_iter": 5},
    LinfPGDAttack: {"rand_init": False, "nb_iter": 5},
    MomentumIterativeAttack: {"nb_iter": 5},
    CarliniWagnerL2Attack: {"num_classes": NUM_CLASS, "max_iterations": 10},
    ElasticNetL1Attack: {"num_classes": NUM_CLASS, "max_iterations": 10},
    FastFeatureAttack: {"rand_init": False, "nb_iter": 5},
    LBFGSAttack: {"num_classes": NUM_CLASS},
    JacobianSaliencyMapAttack: {"num_classes": NUM_CLASS, "gamma": 0.01},
    SpatialTransformAttack: {"num_classes": NUM_CLASS},
    DDNL2Attack: {"nb_iter": 5},
    SparseL1DescentAttack: {"rand_init": False, "nb_iter": 5},
    L1PGDAttack: {"rand_init": False, "nb_iter": 5},
}


def _run_and_assert_original_data_untouched(adversary, data, label):
    data_clone = data.clone()
    adversary.perturb(data, label)
    assert (data_clone == data).all()

    for Attack in targeted_only_attacks:
        if isinstance(adversary, Attack):
            return

    adversary.perturb(data)
    assert (data_clone == data).all()

    adversary.targeted = True
    adversary.perturb(data, label)
    assert (data_clone == data).all()


def _run_data_model_criterion_label_attack(
        data, label, model, criterion, attack, device):
    model.to(device)
    adversary = attack(
        predict=model, loss_fn=criterion, **attack_kwargs[attack])
    data, label = data.to(device), label.to(device)
    _run_and_assert_original_data_untouched(adversary, data, label)


@pytest.mark.parametrize(
    "device, criterion, att_cls", itertools.product(
        devices, label_criteria,
        set(label_attacks).intersection(general_input_attacks)))
def test_running_label_attacks_on_vec(device, criterion, att_cls):
    _run_data_model_criterion_label_attack(
        vecdata, veclabel, vecmodel, criterion, att_cls, device)


@pytest.mark.parametrize(
    "device, criterion, att_cls", itertools.product(
        devices, label_criteria,
        set(label_attacks).intersection(
            image_only_attacks + general_input_attacks)))
def test_running_label_attacks_on_img(device, criterion, att_cls):
    _run_data_model_criterion_label_attack(
        imgdata, imglabel, imgmodel, criterion, att_cls, device)


def _run_data_model_criterion_feature_attack(
        data, model, criterion, attack, device):
    model.to(device)
    adversary = attack(
        predict=model, loss_fn=criterion, **attack_kwargs[attack])
    guide = data.detach().clone()[torch.randperm(len(data))]
    source, guide = data.to(device), guide.to(device)
    source_clone = source.clone()
    adversary.perturb(source, guide)
    assert (source_clone == source).all()


@pytest.mark.parametrize(
    "device, criterion, att_cls", itertools.product(
        devices, feature_criteria,
        set(feature_attacks).intersection(general_input_attacks)))
def test_running_feature_attacks_on_vec(device, criterion, att_cls):
    _run_data_model_criterion_feature_attack(
        vecdata, vecmodel, criterion, att_cls, device)


@pytest.mark.parametrize(
    "device, criterion, att_cls", itertools.product(
        devices, feature_criteria,
        set(feature_attacks).intersection(
            image_only_attacks + general_input_attacks)))
def test_running_feature_attacks_on_img(device, criterion, att_cls):
    _run_data_model_criterion_feature_attack(
        imgdata, imgmodel, criterion, att_cls, device)


def _run_batch_consistent(data, label, model, att_cls, idx):
    if att_cls in feature_attacks:
        guide = data.detach().clone()[torch.randperm(len(data))]
        data, guide = data.to(cpu), guide.to(cpu)
        label_or_guide = guide
    else:
        label_or_guide = label
    model.to(cpu)
    data, label_or_guide = data.to(cpu), label_or_guide.to(cpu)
    adversary = att_cls(model, **attack_kwargs[att_cls])
    torch.manual_seed(0)
    a = adversary.perturb(data, label_or_guide)[idx:idx + 1]
    torch.manual_seed(0)
    b = adversary.perturb(data[idx:idx + 1], label_or_guide[idx:idx + 1])
    assert torch_allclose(a, b)


@pytest.mark.parametrize(
    "idx, att_cls", itertools.product(
        [0, BATCH_SIZE // 2, BATCH_SIZE - 1], batch_consistent_attacks))
def test_batch_consistent_on_vec(idx, att_cls):
    _run_batch_consistent(vecdata, veclabel, vecmodel, att_cls, idx)


@pytest.mark.parametrize(
    "idx, att_cls", itertools.product(
        [0, BATCH_SIZE // 2, BATCH_SIZE - 1], batch_consistent_attacks))
def test_batch_consistent_on_img(idx, att_cls):
    _run_batch_consistent(imgdata, imglabel, imgmodel, att_cls, idx)


if __name__ == '__main__':
    pass
