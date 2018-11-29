# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

from advertorch.context import ctx_eval
from advertorch.context import ctx_noparamgrad
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.context import get_param_grad_state
from advertorch.context import get_module_training_state
from advertorch.context import set_param_grad_off
from advertorch.utils import torch_allclose
from advertorch.test_utils import SimpleModel
from advertorch.test_utils import vecdata


def _generate_models():
    mix_model = SimpleModel()
    mix_model.fc1.training = True
    mix_model.fc2.training = False
    mix_model.fc1.weight.requires_grad = False
    mix_model.fc2.bias.requires_grad = False

    trainon_model = SimpleModel()
    trainon_model.train()

    trainoff_model = SimpleModel()
    trainoff_model.eval()

    gradon_model = SimpleModel()

    gradoff_model = SimpleModel()
    set_param_grad_off(gradoff_model)

    return (
        mix_model, gradon_model, gradoff_model, trainon_model, trainoff_model
    )


mix_model, gradon_model, gradoff_model, trainon_model, trainoff_model = \
    _generate_models()


def _assert_grad_off(module):
    for param in module.parameters():
        assert not param.requires_grad


def _assert_grad_on(module):
    for param in module.parameters():
        assert param.requires_grad


def _assert_training_off(module):
    for mod in module.modules():
        assert not mod.training


def _assert_training_on(module):
    for mod in module.modules():
        assert mod.training


def _run_one_assert_val(ctxmgr, model, assert_inside, assert_outside):
    output = model(vecdata)
    assert_outside(model)
    with ctxmgr(model):
        assert_inside(model)
        assert torch_allclose(output, model(vecdata))
    assert_outside(model)
    assert torch_allclose(output, model(vecdata))


def _run_one_assert_consistent(ctxmgr, model, get_state_fn, assert_inside):
    dct = get_state_fn(mix_model)
    output = model(vecdata)
    with ctxmgr(model):
        assert_inside(model)
        assert torch_allclose(output, model(vecdata))
    newdct = get_state_fn(model)
    assert dct is not newdct
    assert dct == newdct
    assert torch_allclose(output, model(vecdata))


@pytest.mark.parametrize(
    "ctxmgr", (ctx_noparamgrad, ctx_noparamgrad_and_eval))
def test_noparamgrad(ctxmgr):
    _run_one_assert_consistent(ctxmgr, mix_model,
                               get_state_fn=get_param_grad_state,
                               assert_inside=_assert_grad_off)

    _run_one_assert_val(ctxmgr, gradon_model,
                        assert_inside=_assert_grad_off,
                        assert_outside=_assert_grad_on)

    _run_one_assert_val(ctxmgr, gradoff_model,
                        assert_inside=_assert_grad_off,
                        assert_outside=_assert_grad_off)


@pytest.mark.parametrize(
    "ctxmgr", (ctx_eval, ctx_noparamgrad_and_eval))
def test_eval(ctxmgr):
    _run_one_assert_consistent(ctxmgr, mix_model,
                               get_state_fn=get_module_training_state,
                               assert_inside=_assert_training_off)

    _run_one_assert_val(ctxmgr, trainon_model,
                        assert_inside=_assert_training_off,
                        assert_outside=_assert_training_on)

    _run_one_assert_val(ctxmgr, trainoff_model,
                        assert_inside=_assert_training_off,
                        assert_outside=_assert_training_off)


if __name__ == '__main__':
    pass
