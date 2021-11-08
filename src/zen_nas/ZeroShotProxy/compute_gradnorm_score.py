'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

compute gradient norm score
'''
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    input_ = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input_ = input_.cuda(gpu)
    output = model(input_)
    # y_true = torch.rand(size=[batch_size, output.shape[1]], device=torch.device('cuda:{}'.format(gpu))) + 1e-10
    # y_true = y_true / torch.sum(y_true, dim=1, keepdim=True)

    num_classes = output.shape[1]
    label = torch.randint(low=0, high=num_classes, size=[batch_size])

    one_hot_y = F.one_hot(label, num_classes).float()
    if gpu is not None:
        one_hot_y = one_hot_y.cuda(gpu)

    loss = cross_entropy(output, one_hot_y)
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if hasattr(parameter, 'grad') and parameter.grad is not None:
                norm2_sum += torch.norm(parameter.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm
