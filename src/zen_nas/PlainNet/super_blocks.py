'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

define SuperConvBlock class with different kernel size
'''
# pylint: disable=W0613,not-an-iterable,unsubscriptable-object,too-many-arguments
import os
import sys
import uuid
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import global_utils
    import PlainNet
    from PlainNet import _get_right_parentheses_index_, basic_blocks
except ImportError:
    print('fail to import zen_nas modules')


class PlainNetSuperBlockClass(basic_blocks.PlainNetBasicBlockClass):
    """SuperBlock base class"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sub_layers = sub_layers
        self.no_create = no_create
        self.block_list = None
        self.module_list = None

    def forward(self, input_):
        output = input_
        for block in self.block_list:
            output = block(output)
        return output

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},\
                                        {self.stride},{self.sub_layers})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|{self.in_channels},\
                                        {self.out_channels},{self.stride},{self.sub_layers})'

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution

    # pylint: disable=invalid-name
    def get_FLOPs(self, input_resolution):
        resolution = input_resolution
        flops = 0.0
        for block in self.block_list:
            flops += block.get_FLOPs(resolution)
            resolution = block.get_output_resolution(resolution)
        return flops

    def get_model_size(self):
        model_size = 0.0
        for block in self.block_list:
            model_size += block.get_model_size()
        return model_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        if len(self.block_list) == 0:
            self.out_channels = channels
            return

        self.block_list[0].set_in_channels(channels)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                (isinstance(self.block_list[0], (basic_blocks.ConvKX, basic_blocks.ConvDW))) and \
                isinstance(self.block_list[1], basic_blocks.BN):
            self.block_list[1].set_in_channels(last_channels)

    def encode_structure(self):
        """pack channels and sub_layers with list"""

        return [self.out_channels, self.sub_layers]

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert cls.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        sub_layers = int(param_str_split[3])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   sub_layers=sub_layers, block_name=tmp_block_name, no_create=no_create,
                   **kwargs), struct_str[idx + 1:]


# pylint: disable=invalid-name,too-many-instance-attributes
class SuperConvKXBNRELU(PlainNetSuperBlockClass):
    """SuperConv block"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, kernel_size=None,
                 no_create=False, no_reslink=False, no_BN=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN

        # if self.no_reslink:
        #     print('Warning! {} use no_reslink'.format(str(self)))
        # if self.no_BN:
        #     print('Warning! {} use no_BN'.format(str(self)))

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for _ in range(self.sub_layers):
            if not self.no_BN:
                inner_str = f'ConvKX({last_channels},{self.out_channels},\
                              {self.kernel_size},{current_stride})BN({self.out_channels})RELU({self.out_channels})'
            else:
                inner_str = f'ConvKX({last_channels},{self.out_channels},\
                              {self.kernel_size},{current_stride})RELU({self.out_channels})'
            full_str += inner_str

            last_channels = out_channels
            current_stride = 1

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create,
                                                                 no_reslink=no_reslink, no_BN=no_BN)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def forward_pre_relu(self, input_):
        output = input_
        for block in self.block_list[0:-1]:
            output = block(output)
        return output

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},\
                                        {self.stride},{self.sub_layers})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|in={self.in_channels},out={self.out_channels},\
                                        stride={self.stride},sub_layers={self.sub_layers},\
                                        kernel_size={self.kernel_size})'

    def split(self, split_layer_threshold):
        """return str(self)"""

        return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        """ adjust the number to a specific multiple or range"""

        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)
        new_sub_layers = max(1, round(self.sub_layers * sub_layer_scale))

        return type(self).__name__ + f'({self.in_channels},{new_out_channels},{self.stride},{new_sub_layers})'


class SuperConvK1BNRELU(SuperConvKXBNRELU):
    """kernel size 1x1"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         sub_layers=sub_layers,
                         kernel_size=1,
                         no_create=no_create, **kwargs)


class SuperConvK3BNRELU(SuperConvKXBNRELU):
    """kernel size 3x3"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         sub_layers=sub_layers,
                         kernel_size=3,
                         no_create=no_create, **kwargs)


class SuperConvK5BNRELU(SuperConvKXBNRELU):
    """kernel size 5x5"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         sub_layers=sub_layers,
                         kernel_size=5,
                         no_create=no_create, **kwargs)


class SuperConvK7BNRELU(SuperConvKXBNRELU):
    """"kernel size 7x7"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         sub_layers=sub_layers,
                         kernel_size=7,
                         no_create=no_create, **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    """add different kernel size block to block dict"""

    this_py_file_netblocks_dict = {
        'SuperConvK1BNRELU': SuperConvK1BNRELU,
        'SuperConvK3BNRELU': SuperConvK3BNRELU,
        'SuperConvK5BNRELU': SuperConvK5BNRELU,
        'SuperConvK7BNRELU': SuperConvK7BNRELU,

    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
