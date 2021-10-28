'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

define SuperResIDWEXKX class with different parameters
'''
# pylint: disable=invalid-name,too-many-branches,too-many-statements
import os
import sys
import uuid
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import PlainNet
    from PlainNet import _get_right_parentheses_index_
    from PlainNet.super_blocks import PlainNetSuperBlockClass
    import global_utils
except ImportError:
    print('fail to import zen_nas modules')


# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
class SuperResIDWEXKX(PlainNetSuperBlockClass):

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None, sub_layers=None,
                 kernel_size=None, expension=None,
                 no_create=False, no_reslink=False, no_BN=False, use_se=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.expension = expension
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN

        self.use_se = use_se
        if self.use_se:
            print('---debug use_se in ' + str(self))

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for i in range(self.sub_layers):
            inner_str = ''
            # first DW
            dw_channels = global_utils.smart_round(self.bottleneck_channels * self.expension, base=8)
            inner_str += f'ConvKX({last_channels},{dw_channels},{1},{1})'
            if not self.no_BN:
                inner_str += f'BN({dw_channels})'
            inner_str += f'RELU({dw_channels})'

            inner_str += f'ConvDW({dw_channels},{self.kernel_size},{current_stride})'
            if not self.no_BN:
                inner_str += f'BN({dw_channels})'
            inner_str += f'RELU({dw_channels})'
            if self.use_se:
                inner_str += f'SE({dw_channels})'

            inner_str += f'ConvKX({dw_channels},{bottleneck_channels},{1},{1})'
            if not self.no_BN:
                inner_str += f'BN({bottleneck_channels})'
            # inner_str += 'RELU({})'.format(bottleneck_channels)

            if not self.no_reslink:
                if i == 0:
                    res_str = f'ResBlockProj({inner_str})RELU({self.out_channels})'
                else:
                    res_str = f'ResBlock({inner_str})RELU({self.out_channels})'

            else:
                res_str = f'{inner_str}RELU({self.out_channels})'

            full_str += res_str

            # second DW
            inner_str = ''
            dw_channels = global_utils.smart_round(self.out_channels * self.expension, base=8)
            inner_str += f'ConvKX({bottleneck_channels},{dw_channels},{1},{1})'
            if not self.no_BN:
                inner_str += f'BN({dw_channels})'
            inner_str += f'RELU({dw_channels})'

            inner_str += f'ConvDW({dw_channels},{self.kernel_size},{1})'
            if not self.no_BN:
                inner_str += f'BN({dw_channels})'
            inner_str += f'RELU({dw_channels})'
            if self.use_se:
                inner_str += f'SE({dw_channels})'

            inner_str += f'ConvKX({dw_channels},{self.out_channels},{1},{1})'
            if not self.no_BN:
                inner_str += f'BN({self.out_channels})'

            if not self.no_reslink:
                res_str = f'ResBlock({inner_str})RELU({self.out_channels})'
            else:
                res_str = f'{inner_str}RELU({self.out_channels})'

            full_str += res_str
            last_channels = out_channels
            current_stride = 1

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create,
                                                                 no_reslink=no_reslink, no_BN=no_BN, **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},{self.stride},\
                                        {self.bottleneck_channels},{self.sub_layers})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|in={self.in_channels},out={self.out_channels},\
                                        stride={self.stride},btl_channels={self.bottleneck_channels},\
                                        sub_layers={self.sub_layers},kernel_size={self.kernel_size})'

    def encode_structure(self):
        """pack channels and sub_layers to a list"""

        return [self.out_channels, self.sub_layers, self.bottleneck_channels]

    def split(self, split_layer_threshold):
        """split the layer when exceeding threshold"""

        if self.sub_layers >= split_layer_threshold:
            new_sublayers_1 = split_layer_threshold // 2
            new_sublayers_2 = self.sub_layers - new_sublayers_1
            new_block_str1 = type(self).__name__ + f'({self.in_channels},{self.out_channels},{self.stride},\
                                                      {self.bottleneck_channels},{new_sublayers_1})'
            new_block_str2 = type(self).__name__ + f'({self.out_channels},{self.out_channels},{1},\
                                                      {self.bottleneck_channels},{new_sublayers_2})'
            return new_block_str1 + new_block_str2
        return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        """ adjust the number to a specific multiple or range"""

        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)
        new_bottleneck_channels = global_utils.smart_round(self.bottleneck_channels * channel_scale)
        new_sub_layers = max(1, round(self.sub_layers * sub_layer_scale))

        return type(self).__name__ + f'({self.in_channels},{new_out_channels},{self.stride},\
                                        {new_bottleneck_channels},{new_sub_layers})'

    # pylint: disable=arguments-differ
    @classmethod
    def create_from_str(cls, struct_str, **kwargs):
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
        bottleneck_channels = int(param_str_split[3])
        sub_layers = int(param_str_split[4])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                   block_name=tmp_block_name, **kwargs), struct_str[idx + 1:]


class SuperResIDWE1K3(SuperResIDWEXKX):
    """kernel size 3x3, expansion 1.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=3, expension=1.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE2K3(SuperResIDWEXKX):
    """kernel size 3x3, expansion 2.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=3, expension=2.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE4K3(SuperResIDWEXKX):
    """kernel size 3x3, expansion 4.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=3, expension=4.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE6K3(SuperResIDWEXKX):
    """kernel size 3x3, expansion 6.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=3, expension=6.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE1K5(SuperResIDWEXKX):
    """kernel size 5x5, expansion 1.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=5, expension=1.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE2K5(SuperResIDWEXKX):
    """kernel size 5x5, expansion 2.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=5, expension=2.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE4K5(SuperResIDWEXKX):
    """kernel size 5x5, expansion 4.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=5, expension=4.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE6K5(SuperResIDWEXKX):
    """kernel size 5x5, expansion 6.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=5, expension=6.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE1K7(SuperResIDWEXKX):
    """kernel size 7x7, expansion 1.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=7, expension=1.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE2K7(SuperResIDWEXKX):
    """kernel size 7x7, expansion 2.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=7, expension=2.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE4K7(SuperResIDWEXKX):
    """kernel size 7x7, expansion 4.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=7, expension=4.0,
                         no_create=no_create, **kwargs)


class SuperResIDWE6K7(SuperResIDWEXKX):
    """kernel size 7x7, expansion 6.0"""

    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=7, expension=6.0,
                         no_create=no_create, **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    """add different kernel size block to block dict"""

    this_py_file_netblocks_dict = {
        'SuperResIDWE1K3': SuperResIDWE1K3,
        'SuperResIDWE2K3': SuperResIDWE2K3,
        'SuperResIDWE4K3': SuperResIDWE4K3,
        'SuperResIDWE6K3': SuperResIDWE6K3,
        'SuperResIDWE1K5': SuperResIDWE1K5,
        'SuperResIDWE2K5': SuperResIDWE2K5,
        'SuperResIDWE4K5': SuperResIDWE4K5,
        'SuperResIDWE6K5': SuperResIDWE6K5,
        'SuperResIDWE1K7': SuperResIDWE1K7,
        'SuperResIDWE2K7': SuperResIDWE2K7,
        'SuperResIDWE4K7': SuperResIDWE4K7,
        'SuperResIDWE6K7': SuperResIDWE6K7,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
