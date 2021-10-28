'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

"""compute gradient norm score"""

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def network_weight_gaussian_init(net: nn.Module):
    """gaussian initialization"""
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def cross_entropy(logit, target):
    """compute cross entropy loss"""
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def compute_nas_score(gpu, model, resolution, batch_size):
    """compute gradient norm score"""
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)
    output = model(input)
    # y_true = torch.rand(size=[batch_size, output.shape[1]], device=torch.device('cuda:{}'.format(gpu))) + 1e-10
    # y_true = y_true / torch.sum(y_true, dim=1, keepdim=True)

    num_classes = output.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size])

    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu is not None:
        one_hot_y = one_hot_y.cuda(gpu)

    loss = cross_entropy(output, one_hot_y)
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm
