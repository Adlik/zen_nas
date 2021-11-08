# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests using ModelLoader modules to build model """
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.zen_nas import global_utils, ModelLoader
except ImportError:
    print('fail to import ModelLoader')


if __name__ == '__main__':
    opt = global_utils.parse_cmd_options(sys.argv)

    model = ModelLoader.get_model(opt, sys.argv)

    model = model.cuda(opt.gpu)
