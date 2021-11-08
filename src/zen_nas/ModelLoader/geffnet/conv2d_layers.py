""" Conv2D w/ SAME padding, CondConv, MixedConv

A collection of conv layers and padding helpers needed by EfficientNet, MixNet, and
MobileNetV3 models that maintain weight compatibility with original Tensorflow models.

Copyright 2020 Ross Wightman
"""
# pylint: disable=W0613,W0401,too-many-arguments
import os
import sys
from itertools import repeat
from functools import partial
from typing import Tuple, Optional
import math
import collections.abc as container_abcs
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from torch._six import container_abcs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import is_exportable, is_scriptable
except ImportError:
    print('fail to import config')


def _ntuple(num):
    """return tuple containing n elements"""
    def parse(element):
        if isinstance(element, container_abcs.Iterable):
            return element
        return tuple(repeat(element, num))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    """determine whether it is static padding"""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# OW = (W + 2P - K) / S + 1   output = input
def _get_padding(kernel_size, stride=1, dilation=1, **_):
    """get padding value"""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# according above method calculate， but input / stride = output
def _calc_same_pad(input_: int, kernel: int, stride: int, dilation: int):
    """calculate same padding"""
    return max((-(input_ // -stride) - 1) * stride + (kernel - 1) * dilation + 1 - input_, 0)


def _same_pad_arg(input_size, kernel_size, stride, dilation):
    input_height, input_width = input_size
    kernel_height, kernel_width = kernel_size
    pad_h = _calc_same_pad(input_height, kernel_height, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_width, kernel_width, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def _split_channels(num_chan, num_groups):
    """get the number of channels in each group"""
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def conv2d_same(
        input_, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    """conv2d with same padding"""
    input_height, input_width = input_.size()[-2:]
    kernel_height, kernel_width = weight.size()[-2:]
    pad_h = _calc_same_pad(input_height, kernel_height, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_width, kernel_width, stride[1], dilation[1])
    input_ = F.pad(input_, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    # set stride with 0, as x has paded
    return F.conv2d(input_, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    # pylint: disable=arguments-renamed
    def forward(self, input_):
        return conv2d_same(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    # pylint: disable=arguments-renamed
    def forward(self, input_):
        input_size = input_.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            input_ = self.pad(input_)
        return F.conv2d(
            input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_padding_value(padding, kernel_size, **kwargs):
    """get padding value"""
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    """create padding conv2d"""
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if is_exportable():
            assert not is_scriptable()
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    # pylint: disable=too-many-locals
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    # pylint: disable=arguments-differ
    def forward(self, input_):
        input_split = torch.split(input_, self.splits, 1)
        input_out = [conv(input_split[i]) for i, conv in enumerate(self.values())]
        input_ = torch.cat(input_out, 1)
        return input_


def get_condconv_initializer(initializer, num_experts, expert_shape):
    """return condconv initializer"""
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


# pylint: disable=too-many-instance-attributes
class CondConv2d(nn.Module):
    """ Conditional Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = _pair(padding_val)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for per_channel_num in self.weight_shape:
            weight_num_param *= per_channel_num
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """reset parameters"""
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, input_, routing_weights):
        batch, channel, height, width = input_.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (batch * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(batch * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        input_ = input_.view(1, batch * channel, height, width)
        if self.dynamic_padding:
            out = conv2d_same(
                input_, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * batch)
        else:
            out = F.conv2d(
                input_, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * batch)
        out = out.permute([1, 0, 2, 3]).view(batch, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    """select conv2d"""
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        out = MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            out = CondConv2d(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
        else:
            out = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return out
