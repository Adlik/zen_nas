# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zen_nas import ZenNet, global_utils, ModelLoader

if __name__ == '__main__':
    opt = global_utils.parse_cmd_options(sys.argv)

    model = ModelLoader.get_model(opt, argv)

    model = model.cuda(opt.gpu)
