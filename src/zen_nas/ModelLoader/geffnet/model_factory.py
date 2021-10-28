"""model factory"""

from .config import set_layer_config
from .helpers import load_checkpoint

from .gen_efficientnet import *
from .mobilenetv3 import *


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
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
