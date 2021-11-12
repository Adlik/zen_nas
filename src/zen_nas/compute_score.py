# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import Masternet
    from evolution_search import compute_nas_score
except ImportError:
    print('fail to import Masternet, evolution_search')


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--zero_shot_score', type=str, default='Zen',
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--plain_structure', type=str, default=None,
                        help='the text file with model structure str')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


PATH = './ZenNet'

if __name__ == '__main__':
    opt = parse_cmd_options(sys.argv)

    gpu = opt.gpu

    model_plainnet_str_txt = os.path.join(PATH, opt.plain_structure)
    with open(model_plainnet_str_txt, 'r', encoding='utf8') as fid:
        model_plainnet_str = fid.readline().strip()

    Any_Plain_Net = Masternet.MasterNet

    the_nas_score = compute_nas_score(Any_Plain_Net, model_plainnet_str, gpu, opt)

    print(f'zen-score={the_nas_score:.4g}')
