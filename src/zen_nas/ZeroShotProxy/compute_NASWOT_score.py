'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

The implementation of NASWOT score is modified from:
https://github.com/BayesWatch/nas-without-training
'''
# pylint: disable=W0613,invalid-name
import os
import sys
import time
import argparse
import torch
from torch import nn
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from PlainNet import basic_blocks
    import global_utils
    import ModelLoader
except ImportError:
    print('fail to import zen_nas modules')


def network_weight_gaussian_init(net: nn.Module):
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


def logdet(K):
    """get the natural log of the absolute value of the determinant"""
    _, ld = np.linalg.slogdet(K)
    return ld


def get_batch_jacobian(net, x):
    """get jacobian and net's output"""
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach()
    return jacob, y.detach()


def compute_nas_score(gpu, model, resolution, batch_size):
    """compute NASWOT score"""
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input_ = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input_ = input_.cuda(gpu)

    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            print('---- error on model : ')
            print(model)
            raise err

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for _, module in model.named_modules():
        # if 'ReLU' in str(type(module)):
        if isinstance(module, basic_blocks.RELU):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    x = input_
    # jacobs, y = get_batch_jacobian(model, x)
    _, _ = get_batch_jacobian(model, x)

    score = logdet(model.K)

    return float(score)


def parse_cmd_options(argv):
    """parse command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)

    start_timer = time.time()

    for repeat_count in range(args.repeat_times):
        the_score = compute_nas_score(gpu=args.gpu, model=the_model,
                                      resolution=args.input_image_size, batch_size=args.batch_size)

    time_cost = (time.time() - start_timer) / args.repeat_times

    print(f'NASWOT={the_score:.4g}, time cost={time_cost:.4g} second(s)')
