'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

This file is modified from:
https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
'''
# pylint: disable=W0613,too-many-instance-attributes,too-many-arguments
import argparse
import torch
from torch import nn
# from .utils import load_state_dict_from_url
from torch.nn import functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """resnet basicblock definition"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, no_reslink=None):
        super().__init__()
        self.no_reslink = no_reslink
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input_):
        identity = input_

        out = self.conv1(input_)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(input_)

        if not self.no_reslink:
            out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """resnet bottleneck definition"""
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, no_reslink=None):
        super().__init__()
        self.no_reslink = no_reslink

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input_):
        identity = input_

        out = self.conv1(input_)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(input_)

        if not self.no_reslink:
            out += identity
        out = self.relu(out)

        return out


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


class ResNet(nn.Module):
    """ResNet definition"""

    # pylint: disable=too-many-branches,too-many-statements
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_rate=None, no_reslink=None, opt=None, argv=None):
        super().__init__()

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if dropout_rate is None:
            if opt is not None and hasattr(opt, 'dropout_rate'):
                dropout_rate = opt.dropout_rate
            else:
                if module_opt is not None:
                    dropout_rate = module_opt.dropout_rate
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            print(f'!!! Warning !!! {str(type(self))} use dropout_rate={self.dropout_rate:4g}')

        if no_reslink is None:
            if opt is not None and hasattr(opt, 'no_reslink'):
                no_reslink = opt.no_reslink
            else:
                if module_opt is not None:
                    no_reslink = module_opt.no_reslink
        self.no_reslink = no_reslink
        if self.no_reslink:
            print(f'!!! Warning !!! {str(type(self))} use no_reslink')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None \
                               or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], no_reslink=self.no_reslink)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       no_reslink=self.no_reslink)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       no_reslink=self.no_reslink)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       no_reslink=self.no_reslink)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.full_connected = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, no_reslink=False):
        """make network stages"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        if no_reslink:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, no_reslink=no_reslink))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, no_reslink=no_reslink))

        return nn.Sequential(*layers)

    def forward_pre_GAP(self, input_):  # pylint: disable=invalid-name
        """forward before global average pool layer"""
        input_ = self.conv1(input_)
        input_ = self.bn1(input_)
        input_ = self.relu(input_)
        input_ = self.maxpool(input_)

        input_ = self.layer1(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 1.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        input_ = self.layer2(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 2.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)
        input_ = self.layer3(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 3.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        input_ = self.layer4(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 4.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        return input_

    def _forward(self, input_):
        input_ = self.conv1(input_)
        input_ = self.bn1(input_)
        input_ = self.relu(input_)
        input_ = self.maxpool(input_)

        input_ = self.layer1(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 1.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        input_ = self.layer2(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 2.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)
        input_ = self.layer3(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 3.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        input_ = self.layer4(input_)

        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 4.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        input_ = self.avgpool(input_)
        input_ = torch.flatten(input_, 1)
        input_ = self.full_connected(input_)

        return input_

    # Allow for accessing forward method in a inherited class
    forward = _forward

    def extract_stage_features_and_logit(self, input_, target_downsample_ratio=16):
        """extract stage according to downsample ratio

            :param x: network input
            :param target_downsample_ratio (int): target downsample ratio
            :return stage_features, logit
        """

        stage_features_list = []
        image_size = input_.shape[2]

        input_ = self.conv1(input_)
        input_ = self.bn1(input_)
        input_ = self.relu(input_)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.maxpool(input_)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.layer1(input_)
        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 1.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.layer2(input_)
        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 2.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.layer3(input_)
        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 3.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.layer4(input_)
        if self.dropout_rate is not None:
            the_dropout_rate = self.dropout_rate / 4.0 * 4.0
            input_ = F.dropout(input_, p=the_dropout_rate, training=self.training, inplace=False)

        dowsample_ratio = round(image_size / input_.shape[2])
        if dowsample_ratio == target_downsample_ratio:
            stage_features_list.append(input_)
            target_downsample_ratio *= 2

        input_ = self.avgpool(input_)
        input_ = torch.flatten(input_, 1)
        logit = self.full_connected(input_)
        return stage_features_list, logit


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    """create different resnet """
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
