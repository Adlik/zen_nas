# pylint: disable=W0401,import-error
import os
import sys
from helpers import load_checkpoint
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from gen_efficientnet import *
    from mobilenetv3 import *
except ImportError:
    print('fail to import zen_nas modules')


def create_model(
        model_name='mnasnet_100',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):
    """ create a model based on model name

        :param model_name (str): model name
        :param pretrained (bool): pretrained model
        :param num_classes (int): class number
        :param in_chans (int): input channels
        :param checkpoint_path (str): checkpoint file path
        :return model
    """

    model_kwargs = dict(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained, **kwargs)

    if model_name in globals():
        create_fn = globals()[model_name]
        model = create_fn(**model_kwargs)
    else:
        raise RuntimeError(f'Unknown model ({model_name})')

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
