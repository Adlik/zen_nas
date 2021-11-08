'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import os
import sys
import time
import argparse
import torch
from torch import nn
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import global_utils
    import ModelLoader
except ImportError:
    print('fail to import zen_nas modules')


def network_weight_gaussian_init(net: nn.Module):
    """gaussian initialization"""
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                continue

    return net


# pylint: disable=too-many-locals,too-many-arguments
def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    """compute zen score

        :param gpu (int): gpu index
        :param model: model
        :param mixup_gamma (double): mixing coefficient
        :param resolution (int): input image resolution
        :param batch_size (int): batch size
        :param repeat (int): repeat time
        :oaram fp16 (bool): whether using half precision
        :return dict(score)
    """

    score_info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
        model = model.half()
    else:
        dtype = torch.float32

    with torch.no_grad():
        for _ in range(repeat):
            network_weight_gaussian_init(model)
            input1 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input1 + mixup_gamma * input2
            output = model.forward_pre_GAP(input1)
            mixup_output = model.forward_pre_GAP(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(module.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    score_info['avg_nas_score'] = float(avg_nas_score)
    score_info['std_nas_score'] = float(std_nas_score)
    score_info['avg_precision'] = float(avg_precision)
    return score_info


def parse_cmd_options(argv):
    """parse command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)

    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size,
                             repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
