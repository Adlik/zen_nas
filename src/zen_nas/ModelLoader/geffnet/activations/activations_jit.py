""" Activations (jit)

A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.

Copyright 2020 Ross Wightman
"""
# pylint: disable=W0613,no-self-use
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['swish_jit', 'SwishJit', 'mish_jit', 'MishJit',
           'hard_sigmoid_jit', 'HardSigmoidJit', 'hard_swish_jit', 'HardSwishJit']


@torch.jit.script
def swish_jit(input_, inplace: bool = False):
    """Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).

    TODO Rename to SiLU with addition to PyTorch
    """
    return input_.mul(input_.sigmoid())


@torch.jit.script
def mish_jit(input_, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return input_.mul(F.softplus(input_).tanh())


class SwishJit(nn.Module):
    """jit-scripted swish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return swish_jit(input_)


class MishJit(nn.Module):
    """jit-scripted mish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return mish_jit(input_)


@torch.jit.script
def hard_sigmoid_jit(input_, inplace: bool = False):
    """jit-scripted hard sigmoid function"""
    # return F.relu6(x + 3.) / 6.
    return (input_ + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Module):
    """jit-scripted hard sigmoid module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return hard_sigmoid_jit(input_)


@torch.jit.script
def hard_swish_jit(input_, inplace: bool = False):
    """jit-scripted hard swish function"""
    # return x * (F.relu6(x + 3.) / 6)
    return input_ * (input_ + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Module):
    """jit-scripted hard swish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, input_):
        return hard_swish_jit(input_)
