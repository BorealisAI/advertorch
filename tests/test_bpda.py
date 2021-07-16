# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools

import pytest
import torch
import torch.nn as nn

from advertorch.bpda import BPDAWrapper
from advertorch.utils import torch_allclose
from advertorch.test_utils import withgrad_defenses
from advertorch.test_utils import nograd_defenses
from advertorch.test_utils import defense_kwargs
from advertorch.test_utils import defense_data
from advertorch.test_utils import vecdata


cuda = "cuda"
cpu = "cpu"
devices = (cpu, cuda) if torch.cuda.is_available() else (cpu, )


def _identity(x):
    return x


def _straight_through_backward(grad_output, x):
    return grad_output


def _calc_datagrad_on_defense(defense, data):
    data = data.detach().clone().requires_grad_()
    loss = defense(data).sum()
    loss.backward()
    return data.grad.detach().clone()


@pytest.mark.parametrize(
    "device, def_cls", itertools.product(devices, nograd_defenses))
def test_bpda_on_nograd_defense(device, def_cls):
    defense = def_cls(**defense_kwargs[def_cls])

    defense = BPDAWrapper(defense, forwardsub=_identity)
    _calc_datagrad_on_defense(defense, defense_data[def_cls])

    defense = BPDAWrapper(defense, backward=_straight_through_backward)
    _calc_datagrad_on_defense(defense, defense_data[def_cls])


@pytest.mark.parametrize(
    "device, def_cls", itertools.product(devices, withgrad_defenses))
def test_bpda_on_withgrad_defense(device, def_cls):
    defense = def_cls(**defense_kwargs[def_cls])

    grad_from_self = _calc_datagrad_on_defense(
        defense, defense_data[def_cls])

    defense_with_idenity_backward = BPDAWrapper(defense, forwardsub=_identity)
    grad_from_identity_backward = _calc_datagrad_on_defense(
        defense_with_idenity_backward, defense_data[def_cls])

    defense_with_self_backward = BPDAWrapper(defense, forwardsub=defense)
    grad_from_self_backward = _calc_datagrad_on_defense(
        defense_with_self_backward, defense_data[def_cls])

    assert not torch_allclose(grad_from_identity_backward, grad_from_self)
    assert torch_allclose(grad_from_self_backward, grad_from_self)


@pytest.mark.parametrize(
    "device, func", itertools.product(
        devices, [torch.sigmoid, torch.tanh, torch.relu]))
def test_bpda_on_activations(device, func):
    data = vecdata.detach().clone()
    data = data - data.mean()

    grad_from_self = _calc_datagrad_on_defense(func, data)

    func_with_idenity_backward = BPDAWrapper(func, forwardsub=_identity)
    grad_from_identity_backward = _calc_datagrad_on_defense(
        func_with_idenity_backward, data)

    func_with_self_backward = BPDAWrapper(func, forwardsub=func)
    grad_from_self_backward = _calc_datagrad_on_defense(
        func_with_self_backward, data)

    assert not torch_allclose(grad_from_identity_backward, grad_from_self)
    assert torch_allclose(grad_from_self_backward, grad_from_self)


@pytest.mark.parametrize(
    "device, func", itertools.product(
        devices, [nn.Sigmoid(), nn.Tanh(), nn.ReLU()]))
def test_bpda_nograd_on_multi_input(device, func):

    class MultiInputFunc(nn.Module):
        def forward(self, x, y):
            return 2.0 * x - 1.0 * y

    class DummyNet(nn.Module):
        def __init__(self):
            super(DummyNet, self).__init__()
            self.linear = nn.Linear(1200, 10)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            return self.linear(x)

    bpda = BPDAWrapper(forward=MultiInputFunc())

    with torch.enable_grad():
        x = torch.rand(size=(10, 3, 20, 20), device=device,
                       requires_grad=True)
        y = torch.rand_like(x, requires_grad=True)
        z = bpda(x, y)
        z_ = z.detach().requires_grad_()

    net = nn.Sequential(func, DummyNet())
    net.to(device)

    with torch.enable_grad():
        loss_ = net(z_).sum()
        loss = net(z).sum()
    grad_z, = torch.autograd.grad(loss_, [z_])
    grad_x, grad_y = torch.autograd.grad(loss, [x, y])

    assert torch_allclose(grad_x, grad_z)
    assert torch_allclose(grad_y, grad_z)


if __name__ == '__main__':
    from advertorch.defenses import AverageSmoothing2D
    test_bpda_on_withgrad_defense(cpu, AverageSmoothing2D)
