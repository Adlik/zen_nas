import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import Masternet
from evolution_search import compute_nas_score


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


path = './ZenNet'

if __name__ == '__main__':
    opt = parse_cmd_options(sys.argv)

    gpu = opt.gpu
    
    model_plainnet_str_txt = os.path.join(path, opt.plain_structure)   
    with open(model_plainnet_str_txt, 'r') as fid:
        model_plainnet_str = fid.readline().strip()
    
    AnyPlainNet = Masternet.MasterNet

    the_nas_score = compute_nas_score(AnyPlainNet, model_plainnet_str, gpu, opt)

    print(f'zen-score={the_nas_score:.4g}')
