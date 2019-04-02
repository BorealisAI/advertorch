# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# BPDA stands for Backward Pass Differentiable Approximation
# See:
# Athalye, A., Carlini, N. & Wagner, D.. (2018). Obfuscated Gradients Give a
# False Sense of Security: Circumventing Defenses to Adversarial Examples.
# Proceedings of the 35th International Conference on Machine Learning,
# in PMLR 80:274-283

import torch
from torch import autograd
import torch.nn as nn


def _wrap_as_staticmethod(old_func):
    # deprecated for now
    def new_func(ctx, *args, **kwargs):
        return old_func(*args, **kwargs)
    return staticmethod(new_func)


def _wrap_forward_as_function_forward(forward):
    def function_forward(ctx, x):
        ctx.save_for_backward(x)
        return forward(x)
    return staticmethod(function_forward)


def _wrap_backward_as_function_backward(backward):
    def function_backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return backward(grad_output, x)
    return staticmethod(function_backward)


def _create_identity_function():
    def identity(grad_output, x):
        return grad_output
    return identity


def _create_backward_from_forwardsub(forwardsub):
    def backward(grad_output, x):
        # TODO: maybe the detach().clone() pattern can probably be simplified
        x = x.detach().clone().requires_grad_()
        grad_output = grad_output.detach().clone()
        with torch.enable_grad():
            y = forwardsub(x)
            # both of the following return function seems working fine,
            #   using the lower one to make sure there's no side effects
            # return autograd.grad(y, x, grad_output)
            return autograd.grad(y, x, grad_output)[0].detach().clone()
    return backward


class BPDAWrapper(nn.Module):
    """
    Wrap forward module with BPDA backward path
    If forwardsub is not None, then ignore backward

    :param forwardsub: substitute forward function for BPDA
    :param backward: substitute backward function for BPDA
    """

    def __init__(self, forward, forwardsub=None, backward=None):
        """
        Here we assume forward and forwardsub only takes one input x
        and backward takes two inputs grad_output and x
        TODO: adding assert for this, tried inspect.getargspec, but doesn't
            seem to be easy to cover all cases, regular function, class method
            and etc...
        """
        super(BPDAWrapper, self).__init__()

        if forwardsub is not None:
            backward = _create_backward_from_forwardsub(forwardsub)
        else:
            if backward is None:
                backward = _create_identity_function()

        self._create_autograd_function_class()
        self._Function.forward = _wrap_forward_as_function_forward(forward)
        self._Function.backward = _wrap_backward_as_function_backward(backward)

    def forward(self, *args, **kwargs):
        return self._Function.apply(*args, **kwargs)

    def _create_autograd_function_class(self):
        class _Function(autograd.Function):
            pass
        self._Function = _Function
