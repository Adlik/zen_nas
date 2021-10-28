""" EfficientNet / MobileNetV3 Blocks and Builder

Copyright 2020 Ross Wightman
"""
# pylint: disable=W0613,W0401,fixme,too-many-arguments,too-many-locals,too-many-instance-attributes
import os
import sys
import math
import re
from functools import partial
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from geffnet.activations import get_act_layer
    from geffnet.activations.activations import sigmoid
    from conv2d_layers import select_conv2d, CondConv2d, get_condconv_initializer
except ImportError:
    print('fail to import zen_nas modules')

__all__ = ['get_bn_args_tf', 'resolve_bn_args', 'resolve_se_args', 'resolve_act_layer', 'make_divisible',
           'round_channels', 'drop_connect', 'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv',
           'InvertedResidual', 'CondConvResidual', 'EdgeResidual', 'EfficientNetBuilder', 'decode_arch_def',
           'initialize_weight_default', 'initialize_weight_goog', 'BN_MOMENTUM_TF_DEFAULT', 'BN_EPS_TF_DEFAULT'
           ]

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
#
# PyTorch defaults are momentum = .1, eps = 1e-5
#
BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=BN_MOMENTUM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)


def get_bn_args_tf():
    """get tensorflow bn args"""
    return _BN_ARGS_TF.copy()


def resolve_bn_args(kwargs):
    """resolve bn args"""
    bn_args = get_bn_args_tf() if kwargs.pop('bn_tf', False) else {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,  # None == use containing block's activation layer
    reduce_mid=False,
    divisor=1)


def resolve_se_args(kwargs, in_chs, act_layer=None):
    """resolve squeeze and excitation args"""
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, val in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, val)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


def resolve_act_layer(kwargs, default='relu'):
    """resolve activation layer"""
    act_layer = kwargs.pop('act_layer', default)
    if isinstance(act_layer, str):
        act_layer = get_act_layer(act_layer)
    return act_layer


def make_divisible(value: int, divisor: int = 8, min_value: int = None):
    """make number be multiple of divisor"""
    min_value = min_value or divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


def drop_connect(inputs, training: bool = False, drop_connect_rate: float = 0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class SqueezeExcite(nn.Module):
    """Squeeze and Excitation module"""

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1):
        super().__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, input_):
        x_se = input_.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        input_ = input_ * self.gate_fn(x_se)
        return input_


class ConvBnAct(nn.Module):
    """Conv + BN + act_layer"""

    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, pad_type='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super().__init__()
        assert stride in [1, 2]
        norm_kwargs = norm_kwargs or {}
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def forward(self, input_):
        input_ = self.conv(input_)
        input_ = self.bn1(input_)
        input_ = self.act1(input_)
        return input_


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.):
        super().__init__()
        assert stride in [1, 2]
        norm_kwargs = norm_kwargs or {}
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se_block = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se_block = nn.Identity()

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if pw_act else nn.Identity()

    def forward(self, input_):
        residual = input_

        input_ = self.conv_dw(input_)
        input_ = self.bn1(input_)
        input_ = self.act1(input_)

        input_ = self.se_block(input_)

        input_ = self.conv_pw(input_)
        input_ = self.bn2(input_)
        input_ = self.act2(input_)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                input_ = drop_connect(input_, self.training, self.drop_connect_rate)
            input_ += residual
        return input_


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_connect_rate=0.):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs: int = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se_block = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se_block = nn.Identity()  # for jit.script compat

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def forward(self, input_):
        residual = input_

        # Point-wise expansion
        input_ = self.conv_pw(input_)
        input_ = self.bn1(input_)
        input_ = self.act1(input_)

        # Depth-wise convolution
        input_ = self.conv_dw(input_)
        input_ = self.bn2(input_)
        input_ = self.act2(input_)

        # Squeeze-and-excitation
        input_ = self.se_block(input_)

        # Point-wise linear projection
        input_ = self.conv_pwl(input_)
        input_ = self.bn3(input_)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                input_ = drop_connect(input_, self.training, self.drop_connect_rate)
            input_ += residual
        return input_


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 num_experts=0, drop_connect_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super().__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_ratio=se_ratio, se_kwargs=se_kwargs,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs,
            drop_connect_rate=drop_connect_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, input_):
        residual = input_

        # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(input_, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))

        # Point-wise expansion
        input_ = self.conv_pw(input_, routing_weights)
        input_ = self.bn1(input_)
        input_ = self.act1(input_)

        # Depth-wise convolution
        input_ = self.conv_dw(input_, routing_weights)
        input_ = self.bn2(input_)
        input_ = self.act2(input_)

        # Squeeze-and-excitation
        input_ = self.se(input_)

        # Point-wise linear projection
        input_ = self.conv_pwl(input_, routing_weights)
        input_ = self.bn3(input_)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                input_ = drop_connect(input_, self.training, self.drop_connect_rate)
            input_ += residual
        return input_


class EdgeResidual(nn.Module):
    """ EdgeTPU Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0,
                 stride=1, pad_type='', act_layer=nn.ReLU, noskip=False, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        mid_chs = make_divisible(fake_in_chs * exp_ratio) if fake_in_chs > 0 else make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate

        # Expansion convolution
        self.conv_exp = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se_block = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se_block = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, stride=stride, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **norm_kwargs)

    def forward(self, input_):
        residual = input_

        # Expansion convolution
        input_ = self.conv_exp(input_)
        input_ = self.bn1(input_)
        input_ = self.act1(input_)

        # Squeeze-and-excitation
        input_ = self.se_block(input_)

        # Point-wise linear projection
        input_ = self.conv_pwl(input_)
        input_ = self.bn2(input_)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                input_ = drop_connect(input_, self.training, self.drop_connect_rate)
            input_ += residual

        return input_


# pylint: disable=too-few-public-methods
class EfficientNetBuilder:
    """ Build Trunk Blocks for Efficient/Mobile Networks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=None, se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_connect_rate = drop_connect_rate

        # updated during build
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, block_args):
        block_type = block_args.pop('block_type')
        block_args['in_chs'] = self.in_chs
        block_args['out_chs'] = self._round_channels(block_args['out_chs'])
        if 'fake_in_chs' in block_args and block_args['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters for EdgeTPU
            block_args['fake_in_chs'] = self._round_channels(block_args['fake_in_chs'])
        block_args['norm_layer'] = self.norm_layer
        block_args['norm_kwargs'] = self.norm_kwargs
        block_args['pad_type'] = self.pad_type
        # block act fn overrides the model default
        block_args['act_layer'] = block_args['act_layer'] if block_args['act_layer'] is not None else self.act_layer
        assert block_args['act_layer'] is not None
        if block_type == 'ir':
            block_args['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            block_args['se_kwargs'] = self.se_kwargs
            if block_args.get('num_experts', 0) > 0:
                block = CondConvResidual(**block_args)
            else:
                block = InvertedResidual(**block_args)
        elif block_type in ('ds', 'dsa'):
            block_args['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            block_args['se_kwargs'] = self.se_kwargs
            block = DepthwiseSeparableConv(**block_args)
        elif block_type == 'er':
            block_args['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            block_args['se_kwargs'] = self.se_kwargs
            block = EdgeResidual(**block_args)
        elif block_type == 'cn':
            block = ConvBnAct(**block_args)
        else:
            assert False, f'Uknkown block type ({block_type}) while building model.'
        self.in_chs = block_args['out_chs']  # update in_chs for arg of next block
        return block

    def _make_stack(self, stack_args):
        """make one stage containing a list of block"""
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, block_args in enumerate(stack_args):
            if i >= 1:
                # only the first block in any stack can have a stride > 1
                block_args['stride'] = 1
            block = self._make_block(block_args)
            blocks.append(block)
            self.block_idx += 1  # incr global idx (across all stacks)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for _, stack in enumerate(block_args):
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


def _parse_ksize(kernel_size):
    """parse kernel size"""
    if kernel_size.isdigit():
        return int(kernel_size)
    return [int(k) for k in kernel_size.split('.')]


# pylint: disable=too-many-branches
def _decode_block_str(block_str):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op_ in ops:
        # string options being checked on individual basis, combine if they grow
        if op_ == 'noskip':
            noskip = True
        elif op_.startswith('n'):
            # activation fn
            key = op_[0]
            var = op_[1:]
            if var == 're':
                value = get_act_layer('relu')
            elif var == 'r6':
                value = get_act_layer('relu6')
            elif var == 'hs':
                value = get_act_layer('hard_swish')
            elif var == 'sw':
                value = get_act_layer('swish')
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op_)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_layer is None, the model default (passed to model init) will be used
    act_layer = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            noskip=noskip,
        )
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type in ('ds', 'dsa'):
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'er':
        block_args = dict(
            block_type=block_type,
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            fake_in_chs=fake_in_chs,
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            noskip=noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_layer=act_layer,
        )
    else:
        assert False, f'Unknown block type ({block_type})'

    return block_args, num_repeat


def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for repat in repeats[::-1]:
        repeat_scaled = max(1, round((repat / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(repeat_scaled)
        num_repeat -= repat
        num_repeat_scaled -= repeat_scaled
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for block_args, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(block_args) for _ in range(rep)])
    return sa_scaled


def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil', experts_multiplier=1, fix_first_last=False):
    """decode defined architecture string"""
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            block_args, rep = _decode_block_str(block_str)
            if block_args.get('num_experts', 0) > 0 and experts_multiplier > 1:
                block_args['num_experts'] *= experts_multiplier
            stack_args.append(block_args)
            repeats.append(rep)
        if fix_first_last and stack_idx in (0, len(arch_def) - 1):
            arch_args.append(_scale_stage_depth(stack_args, repeats, 1.0, depth_trunc))
        else:
            arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


def initialize_weight_goog(module, name='', fix_group_fanout=True):
    """
      weight init as per Tensorflow Official impl
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    """

    if isinstance(module, CondConv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        if fix_group_fanout:
            fan_out //= module.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), module.num_experts, module.weight_shape)
        init_weight_fn(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        if fix_group_fanout:
            fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        fan_out = module.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in name:
            fan_in = module.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        module.weight.data.uniform_(-init_range, init_range)
        module.bias.data.zero_()


def initialize_weight_default(module, name=''):
    """default initialize weight"""
    if isinstance(module, CondConv2d):
        init_fn = get_condconv_initializer(partial(
            nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu'), module.num_experts, module.weight_shape)
        init_fn(module.weight)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
