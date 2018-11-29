# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools

import pytest
import torch
import torch.nn as nn

from advertorch.test_utils import vecdata
from advertorch.test_utils import vecmodel
from advertorch.test_utils import imgdata
from advertorch.test_utils import imgmodel
from advertorch.test_utils import general_input_defenses
from advertorch.test_utils import image_only_defenses
from advertorch.test_utils import withgrad_defenses
from advertorch.test_utils import nograd_defenses
from advertorch.test_utils import defense_kwargs
from advertorch.test_utils import defense_data

cuda = "cuda"
cpu = "cpu"
devices = (cpu, cuda) if torch.cuda.is_available() else (cpu, )


def _run_data_model_defense(data, model, defense, device):
    defended_model = nn.Sequential(defense, model)
    defended_model.to(device)
    data = data.to(device)
    defended_model(data)


@pytest.mark.parametrize(
    "device, def_cls", itertools.product(devices, general_input_defenses))
def test_running_on_vec(device, def_cls):
    _run_data_model_defense(
        vecdata, vecmodel, def_cls(**defense_kwargs[def_cls]), device)


@pytest.mark.parametrize(
    "device, def_cls",
    itertools.product(devices, general_input_defenses + image_only_defenses))
def test_running_on_img(device, def_cls):
    _run_data_model_defense(
        imgdata, imgmodel, def_cls(**defense_kwargs[def_cls]), device)


@pytest.mark.parametrize(
    "device, def_cls",
    itertools.product(devices, withgrad_defenses))
def test_withgrad(device, def_cls):
    defense = def_cls(**defense_kwargs[def_cls])
    data = defense_data[def_cls]
    data.requires_grad_()
    loss = defense(data).sum()
    loss.backward()


@pytest.mark.parametrize(
    "device, def_cls",
    itertools.product(devices, nograd_defenses))
def test_defenses_nograd(device, def_cls):
    with pytest.raises(NotImplementedError):
        defense = def_cls(**defense_kwargs[def_cls])
        data = defense_data[def_cls]
        data.requires_grad_()
        loss = defense(data).sum()
        loss.backward()


if __name__ == '__main__':
    pass
