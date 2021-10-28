'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

get model FLOPs and parameters
'''
import os
import sys
from ptflops import get_model_complexity_info
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import ModelLoader
    import global_utils
    import ZenNet
except ImportError:
    print('fail to import zen_nas modules')


def main(opt, argv):
    """get model flops and parameters"""
    model = ModelLoader.get_model(opt, argv)
    flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print(f'Flops:  {flops:4g}')
    print(f'Params: {params:4g}')


def get_flops_params(opt):
    """get model flops and parameters"""
    model = ZenNet.get_ZenNet(opt.arch)
    flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print(f'Flops:  {flops:4g}')
    print(f'Params: {params:4g}')


if __name__ == "__main__":
    option = global_utils.parse_cmd_options(sys.argv)

    # get_flops_params(opt)
    main(option, sys.argv)
