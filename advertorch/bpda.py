# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
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
import torch.nn as nn

__all__ = ['BPDAWrapper']


class FunctionWrapper(nn.Module):
    """`nn.Module` wrapping a `torch.autograd.Function`."""

    def __init__(self, func):
        """Wraps the provided function `func`.

        :param func: the `torch.autograd.Function` to be wrapped.
        """
        super(FunctionWrapper, self).__init__()
        self.func = func

    def forward(self, *inputs):
        """Wraps the `forward` method of `func`."""
        return self.func.apply(*inputs)


class BPDAWrapper(FunctionWrapper):
    """Backward Pass Differentiable Approximation.

    The module should be provided a `forward` method and a `backward`
    method that approximates the derivatives of `forward`.

    The `forward` function is called in the forward pass, and the
    `backward` function is used to find gradients in the backward pass.

    The `backward` function can be implicitly provided-by providing
    `forwardsub` - an alternative forward pass function, which its
    gradient will be used in the backward pass.

    If not `backward` nor `forwardsub` are provided, the `backward`
    function will be assumed to be the identity.

    :param forward: `forward(*inputs)` - the forward function for BPDA.
    :param forwardsub: (Optional) a substitute forward function, for the
                       gradients approximation of `forward`.
    :param backward: (Optional) `backward(inputs, grad_outputs)` the
                     backward pass function for BPDA.
    """

    def __init__(self, forward, forwardsub=None, backward=None):
        func = self._create_func(forward, backward, forwardsub)
        super(BPDAWrapper, self).__init__(func)

    @classmethod
    def _create_func(cls, forward_fn, backward_fn, forwardsub_fn):
        if backward_fn is not None:
            return cls._create_func_backward(forward_fn, backward_fn)

        if forwardsub_fn is not None:
            return cls._create_func_forwardsub(forward_fn, forwardsub_fn)

        return cls._create_func_forward_only(forward_fn)

    @classmethod
    def _create_func_forward_only(cls, forward_fn):
        """Creates a differentiable `Function` given the forward function,
        and the identity as backward function."""

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                ctx.save_for_backward(*inputs)
                return forward_fn(*inputs, **kwargs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                inputs = ctx.saved_tensors
                if len(grad_outputs) == len(inputs):
                    return grad_outputs
                elif len(grad_outputs) == 1:
                    return tuple([grad_outputs[0] for _ in inputs])

                raise ValueError("Expected %d gradients but got %d" %
                                 (len(inputs), len(grad_outputs)))


        return Func

    @classmethod
    def _create_func_forwardsub(cls, forward_fn, forwardsub_fn):
        """Creates a differentiable `Function` given the forward function,
        and a substitute forward function.

        The substitute forward function is used to approximate the gradients
        in the backward pass.
        """

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                ctx.save_for_backward(*inputs)
                return forward_fn(*inputs, **kwargs)

            @staticmethod
            @torch.enable_grad()  # enables grad in the method's scope
            def backward(ctx, *grad_outputs):
                inputs = ctx.saved_tensors
                inputs = [x.detach().clone().requires_grad_() for x in inputs]
                outputs = forwardsub_fn(*inputs)
                return torch.autograd.grad(outputs, inputs, grad_outputs)

        return Func

    @classmethod
    def _create_func_backward(cls, forward_fn, backward_fn):
        """Creates a differentiable `Function` given the forward and backward
        functions."""

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                ctx.save_for_backward(*inputs)
                return forward_fn(*inputs, **kwargs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                inputs = ctx.saved_tensors
                return backward_fn(inputs, grad_outputs)

        return Func
