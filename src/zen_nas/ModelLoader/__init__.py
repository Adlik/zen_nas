'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

The geffnet module is modified from:
https://github.com/rwightman/gen-efficientnet-pytorch
'''
# pylint: disable=invalid-name
import os
import sys
import importlib.util
import torchvision.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import PlainNet
    import geffnet
    import myresnet
except ImportError:
    print('fail to import zen_nas modules')


torchvision_model_name_list = sorted(name for name, value in torchvision.models.__dict__.items()
                                     if name.islower() and not name.startswith("__") and callable(value))


def _get_model_(arch, num_classes, pretrained=False, opt=None, argv=None):
    """create arch model and return

        :param arch (str): model
        :param num_classes (int): class number
        :param pretrained (bool): pretrained
        :return arch model
    """

    # load torch vision model
    if arch in torchvision_model_name_list:
        if pretrained:
            print(f'Using pretrained model: {arch}')
            model = torchvision.models.__dict__[arch](pretrained=True, num_classes=num_classes)
        else:
            print(f'Create torchvision model: {arch}')
            model = torchvision.models.__dict__[arch](num_classes=num_classes)

        # my implementation of resnet
    elif arch == 'myresnet18':
        print(f'Create model: {arch}')
        model = myresnet.resnet18(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet34':
        print(f'Create model: {arch}')
        model = myresnet.resnet34(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet50':
        print(f'Create model: {arch}')
        model = myresnet.resnet50(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet101':
        print(f'Create model: {arch}')
        model = myresnet.resnet101(pretrained=False, opt=opt, argv=argv)
    elif arch == 'myresnet152':
        print(f'Create model: {arch}')
        model = myresnet.resnet152(pretrained=False, opt=opt, argv=argv)

    # geffnet
    elif arch.startswith('geffnet_'):
        model_name = arch[len('geffnet_'):]
        model = geffnet.create_model(model_name, pretrained=pretrained)

    # PlainNet
    elif arch == 'PlainNet':
        print(f'Create model: {arch}')
        model = PlainNet.PlainNet(num_classes=num_classes, opt=opt, argv=argv)

    # Any PlainNet
    elif arch.find('.py:MasterNet') >= 0:
        module_path = arch.split(':')[0]
        assert arch.split(':')[1] == 'MasterNet'
        my_working_dir = os.path.dirname(os.path.dirname(__file__))
        module_full_path = os.path.join(my_working_dir, module_path)

        spec = importlib.util.spec_from_file_location('any_plain_net', module_full_path)
        any_plain_net = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(any_plain_net)
        print(f'Create model: {arch}')
        model = any_plain_net.MasterNet(num_classes=num_classes, opt=opt, argv=argv)

    else:
        raise ValueError(f'Unknown model arch: {arch}')

    return model


def get_model(opt, argv):
    """get arch model"""
    return _get_model_(arch=opt.arch, num_classes=opt.num_classes, pretrained=opt.pretrained, opt=opt, argv=argv)
