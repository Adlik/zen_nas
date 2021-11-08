""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Copyright 2020 Ross Wightman
"""
# pylint: disable=W0613
from torch import nn
from torch.nn import functional as F


def swish(input_, inplace: bool = False):
    """Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).

    TODO Rename to SiLU with addition to PyTorch
    """
    return input_.mul_(input_.sigmoid()) if inplace else input_.mul(input_.sigmoid())


class Swish(nn.Module):
    """Swish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return swish(input_, self.inplace)


def mish(input_, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return input_.mul(F.softplus(input_).tanh())


class Mish(nn.Module):
    """Mish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return mish(input_, self.inplace)


def sigmoid(input_, inplace: bool = False):
    """sigmoid function"""
    return input_.sigmoid_() if inplace else input_.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    """sigmoid module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return input_.sigmoid_() if self.inplace else input_.sigmoid()


def tanh(input_, inplace: bool = False):
    """Tanh function"""
    return input_.tanh_() if inplace else input_.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    """Tanh module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return input_.tanh_() if self.inplace else input_.tanh()


def hard_swish(input_, inplace: bool = False):
    """hard swish function"""
    inner = F.relu6(input_ + 3.).div_(6.)
    return input_.mul_(inner) if inplace else input_.mul(inner)


class HardSwish(nn.Module):
    """hard swish module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return hard_swish(input_, self.inplace)


def hard_sigmoid(input_, inplace: bool = False):
    """hard sigmoid function"""
    if inplace:
        return input_.add_(3.).clamp_(0., 6.).div_(6.)
    return F.relu6(input_ + 3.) / 6.


class HardSigmoid(nn.Module):
    """hard sigmoid module"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return hard_sigmoid(input_, self.inplace)
