'''
Copyright 2019 ZTE corporation. All Rights Reserved.
SPDX-License-Identifier: Apache-2.

generate random network and mutated network
'''
# pylint: disable=global-statement
import os
import sys
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import global_utils
    import PlainNet
    from PlainNet import super_blocks, SuperResKXKX, SuperResK1KXK1
except ImportError:
    print('fail to import zen_nas modules')

SEARCH_SPACE = [SuperResKXKX, SuperResK1KXK1]


def get_block_from_module(module, block_dict: dict):
    """collect all optional block from module"""
    assert hasattr(module, 'register_netblocks_dict')
    block_dict = module.register_netblocks_dict(block_dict)
    return block_dict


def register_serach_space(candidate_blocks):
    """register search space"""
    global SEARCH_SPACE
    SEARCH_SPACE.clear()
    SEARCH_SPACE = candidate_blocks.copy()


def get_random_block():
    """get random block from serach space"""
    search_space_block_dict = {}
    for module in SEARCH_SPACE:
        search_space_block_dict = get_block_from_module(module, search_space_block_dict)

    block_list = list(search_space_block_dict.values())
    return np.random.choice(block_list)


def get_random_channels_ratio(search_range: list = None):
    """determin mutated channel ratio"""
    if search_range is not None:
        assert len(search_range) == 2, 'need two number to limit to choose number in ranges'
        search_range.sort()
        return random.uniform(search_range[0], search_range[1])
    candidate = [2.5, 2, 1.5, 1.25, 1, 1 / 1.25, 1 / 1.5, 1 / 2, 1 / 2.5]
    return np.random.choice(candidate)


def get_random_sublayers_change():
    """determin mutated sublayer """
    candidate = [-2, -1, 0, 1, 2]
    return np.random.choice(candidate)


def verify_channels(channel, ratio):
    """restrict range"""
    channel *= ratio
    new_channel = global_utils.smart_round(channel, base=8)
    return min(new_channel, 2048)


def verify_sublayers(sublayers, change: int):
    """restrict range"""
    sublayers += change
    return max(1, sublayers)


def mutated_block(block_list: list, block_id: int):
    """construct mutated block objec"""
    chosen_block = block_list[block_id]
    in_channel = chosen_block.in_channels
    out_channel = chosen_block.out_channels
    stride = chosen_block.stride
    sub_layers = chosen_block.sub_layers
    ratio = get_random_channels_ratio()

    if isinstance(chosen_block, super_blocks.SuperConvKXBNRELU):
        out_channel = verify_channels(out_channel, ratio)
        mutated_block_obj = chosen_block.__class__(in_channel, out_channel, stride, 1)
    else:
        bottleneck_channel = chosen_block.bottleneck_channels
        new_block = get_random_block()
        out_channel = verify_channels(out_channel, ratio)
        bottleneck_channel = verify_channels(bottleneck_channel, ratio)
        # out_channel and bottleneck size too big, CUDNN_STATUS_NOT_SUPPORTED may occur in model training
        bottleneck_channel = min(bottleneck_channel, 512)
        change = get_random_sublayers_change()
        sub_layers = verify_sublayers(sub_layers, change)
        mutated_block_obj = new_block(in_channel, out_channel, stride, bottleneck_channel, sub_layers)

    return mutated_block_obj


def get_mutated_structure_str(any_plain_net, structure_str, num_classes, num_replaces=1):
    """generate mutated network string"""
    the_net = any_plain_net(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)
    selected_random_id_set = set()
    for _ in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)

        new_block_obj = mutated_block(the_net.block_list, random_id)
        if random_id < len(the_net.block_list) - 1:
            out_channel = new_block_obj.out_channels
            the_net.block_list[random_id + 1].set_in_channels(out_channel)
        the_net.block_list[random_id] = new_block_obj

    assert hasattr(the_net, 'split')
    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str


def get_random_initialized_structure_str():
    """generate random network from serach space"""
    struct_str = ""
    input_channel = 3
    out_channel = 32
    str_len = random.randint(6, 10)
    channel_choice = [8 * i for i in range(1, 16)]
    # sub_layer_choice = [1, 2]
    for i in range(str_len):
        if i <= 3 or i == round((3 + str_len - 1) / 2):
            stride = 2
        else:
            stride = 1

        if i == 0:
            struct_str += f'SuperConvK3BNRELU({input_channel},{out_channel},{stride},{1})'
            input_channel = out_channel
        elif i == str_len - 1:
            out_channel = np.random.choice(channel_choice)
            struct_str += f'SuperConvK1BNRELU({input_channel},{out_channel},{stride},{1})'
        else:
            out_channel = np.random.choice(channel_choice)
            bottleneck_channel = np.random.choice(channel_choice)
            # sub_layer = np.random.choice(sub_layer_choice)
            sub_layer = 1

            block = get_random_block()
            struct_str += block.__name__ + f'({input_channel},{out_channel},\
                                     {stride},{bottleneck_channel},{sub_layer})'
            input_channel = out_channel
    return struct_str
