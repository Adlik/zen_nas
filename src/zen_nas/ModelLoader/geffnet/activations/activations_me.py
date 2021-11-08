""" Activations (memory-efficient w/ custom autograd)

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

These activations are not compatible with jit scripting or ONNX export of the model, please use either
the JIT or basic versions of the activations.

Copyright 2020 Ross Wightman
"""
# pylint: disable=W0613,arguments-differ,no-self-use,abstract-method
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['swish_me', 'SwishMe', 'mish_me', 'MishMe',
           'hard_sigmoid_me', 'HardSigmoidMe', 'hard_swish_me', 'HardSwishMe']


@torch.jit.script
def swish_jit_fwd(input_):
    """jit-scripted swish forward"""
    return input_.mul(torch.sigmoid(input_))


@torch.jit.script
def swish_jit_bwd(input_, grad_output):
    """jit-scripted swish backward"""
    input_sigmoid = torch.sigmoid(input_)
    return grad_output * (input_sigmoid * (1 + input_ * (1 - input_sigmoid)))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200

    Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).

    TODO Rename to SiLU with addition to PyTorch
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return swish_jit_bwd(output, grad_output)


def swish_me(input_, inplace=False):
    """jit-scripted swish_me function"""
    return SwishJitAutoFn.apply(input_)


class SwishMe(nn.Module):
    """jit-scripted swish_me module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return SwishJitAutoFn.apply(input_)


@torch.jit.script
def mish_jit_fwd(input_):
    """jit-scripted mish forward"""
    return input_.mul(torch.tanh(F.softplus(input_)))


@torch.jit.script
def mish_jit_bwd(output, grad_output):
    """jit-scripted mish backward"""
    output_sigmoid = torch.sigmoid(output)
    output_tanh_sp = F.softplus(output).tanh()
    return grad_output.mul(output_tanh_sp + output * output_sigmoid * (1 - output_tanh_sp * output_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return mish_jit_bwd(output, grad_output)


def mish_me(input_, inplace=False):
    """jit-scripted mish_me function"""
    return MishJitAutoFn.apply(input_)


class MishMe(nn.Module):
    """jit-scripted mish_me module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return MishJitAutoFn.apply(input_)


@torch.jit.script
def hard_sigmoid_jit_fwd(input_, inplace: bool = False):
    """jit-scripted hard sigmoid forward"""
    return (input_ + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_sigmoid_jit_bwd(output, grad_output):
    """git-scripted hard sigmoid backward"""
    tmp = torch.ones_like(output) * ((output >= -3.) & (output <= 3.)) / 6.
    return grad_output * tmp


class HardSigmoidJitAutoFn(torch.autograd.Function):
    """jit-scripted hard sigmoid module"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(output, grad_output)


def hard_sigmoid_me(input_, inplace: bool = False):
    """jit-scripted hard_sigmoid_me function"""
    return HardSigmoidJitAutoFn.apply(input_)


class HardSigmoidMe(nn.Module):
    """jit-scripted hard_sigmoid_me module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return HardSigmoidJitAutoFn.apply(input_)


@torch.jit.script
def hard_swish_jit_fwd(input_):
    """jit-scripted hard swish forward"""
    return input_ * (input_ + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_swish_jit_bwd(output, grad_output):
    """jit-scripted hard swish backward"""
    tmp = torch.ones_like(output) * (output >= 3.)
    tmp = torch.where((output >= -3.) & (output <= 3.), output / 3. + .5, tmp)
    return grad_output * tmp


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(output, grad_output)


def hard_swish_me(input_, inplace=False):
    """jit-scripted hard_swish_me function"""
    return HardSwishJitAutoFn.apply(input_)


class HardSwishMe(nn.Module):
    """jit-scripted hard_swish_me module"""

    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return HardSwishJitAutoFn.apply(input_)
