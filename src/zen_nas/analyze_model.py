'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

"""get model FLOPs and parameters"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import ModelLoader
import global_utils
from ptflops import get_model_complexity_info
import ZenNet


def main(opt, argv):
    """get model flops and parameters"""
    model = ModelLoader.get_model(opt, argv)
    flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print('Flops:  {:4g}'.format(flops))
    print('Params: {:4g}'.format(params))


def get_flops_params(opt):
    """get model flops and parameters"""
    model = ZenNet.get_ZenNet(opt.arch)
    flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print('Flops:  {:4g}'.format(flops))
    print('Params: {:4g}'.format(params))


if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    
    # get_flops_params(opt)
    main(opt, sys.argv)
    