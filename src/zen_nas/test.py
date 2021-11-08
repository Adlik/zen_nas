'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

Usage:
python val.py --gpu 0 --arch zennet_imagenet1k_latency02ms_res192
'''
# pylint: disable=invalid-name
import os
import sys
import argparse
import math
import PIL
import torch
from torchvision import transforms, datasets
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import ZenNet
except ImportError:
    print('fail to import ZenNet')

imagenet_data_dir = os.path.expanduser('~/data/imagenet/imagenet-torch/')


def accuracy(output_, target_, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target_.size(0)

        _, pred = output_.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=6,
                        help='number of workers to load dataset.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID. None for CPU.')
    parser.add_argument('--data', type=str, default=imagenet_data_dir,
                        help='ImageNet data directory.')
    parser.add_argument('--arch', type=str, default=None,
                        help='model to be evaluated.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--apex', action='store_true',
                        help='Use NVIDIA Apex (float16 precision).')

    opt, _ = parser.parse_known_args(sys.argv)

    if opt.apex:
        # pylint: disable=no-name-in-module
        from apex import amp
    # else:
    #     print('Warning!!! The GENets are trained by NVIDIA Apex, '
    #           'it is suggested to turn on --apex in the evaluation. '
    #           'Otherwise the model accuracy might be harmed.')

    input_image_size = ZenNet.zennet_model_zoo[opt.arch]['resolution']
    crop_image_size = ZenNet.zennet_model_zoo[opt.arch]['crop_image_size']

    print(f'Evaluate {opt.arch} at {input_image_size}x{input_image_size} resolution.')

    # load dataset
    val_dir = os.path.join(opt.data, 'val')
    input_image_crop = 0.875
    resize_image_size = int(math.ceil(crop_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC),
                      transforms.CenterCrop(crop_image_size), transforms.ToTensor(), transforms_normalize]
    transformer = transforms.Compose(transform_list)
    val_dataset = datasets.ImageFolder(val_dir, transformer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.workers, pin_memory=True, sampler=None)

    # load model
    model = ZenNet.get_ZenNet(opt.arch, pretrained=True)
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        torch.backends.cudnn.benchmark = True
        model = model.cuda(opt.gpu)
        print(f'Using GPU {opt.gpu}.')
        if opt.apex:
            model = amp.initialize(model, opt_level="O1")
        elif opt.fp16:
            model = model.half()

    model.eval()
    acc1_sum = 0
    acc5_sum = 0
    num = 0
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):

            if opt.gpu is not None:
                input_ = input_.cuda(opt.gpu, non_blocking=True)
                target = target.cuda(opt.gpu, non_blocking=True)
                if opt.fp16:
                    input_ = input_.half()

            input_ = torch.nn.functional.interpolate(input_, input_image_size, mode='bilinear')
            output = model(input_)
            # pylint: disable=unbalanced-tuple-unpacking
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1_sum += acc1[0] * input_.shape[0]
            acc5_sum += acc5[0] * input_.shape[0]
            num += input_.shape[0]

            if i % 100 == 0:
                print(f'mini_batch {i}, top-1 acc={acc1[0]:4g}%, top-5 acc={acc5[0]:4g}%, \
                      number of evaluated images={num}')

    acc1_avg = acc1_sum / num
    acc5_avg = acc5_sum / num

    print(f'*** arch={opt.arch}, validation top-1 acc={acc1_avg}%, top-5 acc={acc5_avg}%, \
          number of evaluated images={num}')
