'''
Copyright 2021 ZTE corporation. All Rights Reserved.
SPDX-License-Identifier: Apache-2.

Support single GPU, horovod, apex with mixed precision distributed training, teacher distillation
'''
# mypy: ignore-errors
# pylint: disable=not-callable,too-many-lines,too-many-branches,too-many-statements,no-member
import os
import sys
import copy
import time
import logging
from typing import List, Any
import distutils.dir_util
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import horovod.torch as hvd
except ImportError:
    print('fail to import hvd.')

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp  # pylint: disable=no-name-in-module
    # from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print('fail to import apex.')

try:
    import ModelLoader
    import DataLoader
    import global_utils
except ImportError:
    print('fail to import self-defined modules')


def get_logger(log_filename=None, level=logging.INFO):
    """get log object, support distributed training """
    logger = logging.getLogger()
    if log_filename is not None:
        distutils.dir_util.mkpath(os.path.dirname(log_filename))
        logger.addHandler(
            logging.FileHandler(log_filename)
        )
    logger.addHandler(
        logging.StreamHandler()
    )
    logger.setLevel(level)
    return logger


def ts_feature_map_loss(output, label):
    """compute smooth L1 loss"""
    return torch.nn.functional.smooth_l1_loss(output, label)


# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
class TeacherStudentDistillNet(nn.Module):
    """warp teacher and student model, implement the distillation process"""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, opt, argv, target_downsample_ratio=None):
        super().__init__()
        self.opt = opt
        self.argv = argv
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.target_downsample_ratio = target_downsample_ratio

        assert hasattr(self.teacher_model, 'extract_stage_features_and_logit')
        assert hasattr(self.student_model, 'extract_stage_features_and_logit')

        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.student_model.eval()

        # create project layer
        teacher_test_img = torch.randn((1, 3, opt.teacher_input_image_size, opt.teacher_input_image_size))
        teacher_stage_features_list, _ = self.teacher_model.extract_stage_features_and_logit(
            teacher_test_img, target_downsample_ratio=target_downsample_ratio)
        student_test_img = torch.randn((1, 3, opt.input_image_size, opt.input_image_size))
        student_stage_features_list, _ = self.student_model.extract_stage_features_and_logit(
            student_test_img, target_downsample_ratio=target_downsample_ratio)

        assert len(teacher_stage_features_list) == len(student_stage_features_list)

        self.proj_conv_list = nn.ModuleList()
        for teacher_feature, student_feature in zip(teacher_stage_features_list, student_stage_features_list):
            proj_conv_seq_blocks: List[Any] = []
            proj_conv_seq_blocks.append(nn.Conv2d(student_feature.shape[1],
                                                  teacher_feature.shape[1], kernel_size=1, stride=1))

            if not opt.ts_proj_no_bn:
                proj_conv_seq_blocks.append(nn.BatchNorm2d(teacher_feature.shape[1]))
            else:
                print('--- use ts_proj_no_bn')

            if not opt.ts_proj_no_relu:
                proj_conv_seq_blocks.append(nn.ReLU(teacher_feature.shape[1]))
            else:
                print('--- use ts_proj_no_relu')

            proj_conv = nn.Sequential(*proj_conv_seq_blocks)
            self.proj_conv_list.append(proj_conv)

        self.teacher_stage_features_list = None
        self.teacher_logit = None
        self.student_stage_features_list = None
        self.student_logit = None

        # default initialize
        self.init_parameters()

        # bn eps
        for layer in self.student_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3
        for block in self.proj_conv_list:
            for layer in block.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3

    def train(self, mode=True):
        """set to train mode"""
        self.training = True
        self.student_model.train(mode)
        self.teacher_model.eval()

    def eval(self):
        """set to eval mode"""
        self.training = False
        self.student_model.eval()
        self.teacher_model.eval()

    def forward(self, input_):
        """obtain the stage features of the student and teacher and the logit of the student"""
        if self.training:
            if input_.shape[2] != self.opt.teacher_input_image_size:
                teacher_x = F.interpolate(input_, self.opt.teacher_input_image_size, mode='bilinear')
            else:
                teacher_x = input_
            self.teacher_stage_features_list, self.teacher_logit = self.teacher_model.extract_stage_features_and_logit(
                teacher_x, target_downsample_ratio=self.target_downsample_ratio)

        if input_.shape[2] != self.opt.input_image_size:
            student_x = F.interpolate(input_, self.opt.input_image_size, mode='bilinear')
        else:
            student_x = input_
        self.student_stage_features_list, self.student_logit = self.student_model.extract_stage_features_and_logit(
            student_x, target_downsample_ratio=self.target_downsample_ratio)

        return self.student_logit

    def compute_ts_distill_loss(self):
        """compute feature loss and logit loss"""
        feature_loss = 0.0
        for teacher_feature, student_feature, proj_conv in zip(self.teacher_stage_features_list,
                                                               self.student_stage_features_list, self.proj_conv_list):
            if self.opt.ts_clip is not None:
                teacher_feature = torch.clamp(teacher_feature, -1 * self.opt.ts_clip, self.opt.ts_clip)

            if teacher_feature.shape[2] != student_feature.shape[2]:
                teacher_feature = F.interpolate(teacher_feature, student_feature.shape[2], mode='bilinear')

            proj_sf = proj_conv(student_feature)
            # proj_tf = proj_conv(tf)

            # feature_loss = feature_loss + hierarchical_loss(proj_sf, tf, alpha=0.5, k=2)
            feature_loss = feature_loss + ts_feature_map_loss(proj_sf, teacher_feature)
            # feature_loss = feature_loss + hierarchical_loss(sf, proj_tf, alpha=alpha, k=k)

        prob_logit = F.log_softmax(self.student_logit, dim=1)
        target = F.softmax(self.teacher_logit, dim=1)
        logit_loss = -(target * prob_logit).sum(dim=1).mean()

        return feature_loss, logit_loss

    def init_parameters(self):
        """initialize student model and projection function"""
        for block in self.proj_conv_list:
            network_weight_zero_init(block)
        if hasattr(self.student_model, 'init_parameters'):
            self.student_model.init_parameters()
        else:
            print('Warning!!! student model has no init_parameters()!')


def save_checkpoint(checkpoint_filename, state_dict):
    """save model state_dict"""
    save_dir = os.path.dirname(checkpoint_filename)
    base_filename = os.path.basename(checkpoint_filename)
    backup_filename = os.path.join(save_dir, base_filename + '.backup')

    global_utils.mkdir(save_dir)

    if os.path.isfile(checkpoint_filename):
        if os.path.isfile(backup_filename):
            os.remove(backup_filename)
        os.rename(checkpoint_filename, backup_filename)

    torch.save(state_dict, checkpoint_filename)
    if os.path.isfile(backup_filename):
        os.remove(backup_filename)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':4g', disp_avg=True):
        self.name = name
        self.fmt = fmt
        self.disp_avg = disp_avg

        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}{val' + self.fmt + '}'
        fmtstr = fmtstr.format(name=self.name, val=self.val)
        if self.disp_avg:
            fmtstr += '({avg' + self.fmt + '})'
            fmtstr = fmtstr.format(avg=self.avg)
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            decay.append(module.weight)

            if module.bias is not None:
                no_decay.append(module.bias)

        else:
            if hasattr(module, 'weight'):
                no_decay.append(module.weight)
            if hasattr(module, 'bias'):
                no_decay.append(module.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def network_weight_MSRAPrelu_init(net: nn.Module):  # pylint: disable=invalid-name
    """the gain of xavier_normal_ is computed from gain=magnitude * sqrt(3)
       where magnitude is 2/(1+0.25**2). [mxnet implementation]
    """

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=3.26033)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 3.26033 * np.sqrt(2 / (module.weight.shape[0] + module.weight.shape[1])))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            pass

    return net


def network_weight_xavier_init(net: nn.Module):
    """the gain of xavier_normal_ is gotten by nn.init.calculate_gain('relu)"""
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 3.26033 * np.sqrt(2 / (module.weight.shape[0] + module.weight.shape[1])))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            pass

    return net


def network_weight_stupid_init(net: nn.Module):
    """initialize tensor with torch.randn """
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                device = module.weight.device
                in_channels, _, kernel_size1, kernel_size2 = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / \
                    np.sqrt(kernel_size1 * kernel_size2 * in_channels)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                device = module.weight.device
                in_channels, _ = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                continue

    return net


def network_weight_zero_init(net: nn.Module):
    """initialize tensor with torch.randn and scale by 1e-4 """
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                device = module.weight.device
                in_channels, _, kernel_size1, kernek_size2 = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / \
                    np.sqrt(kernel_size1 * kernek_size2 * in_channels) * 1e-4
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                device = module.weight.device
                in_channels, _ = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / np.sqrt(in_channels) * 1e-4
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                continue

    return net


def network_weight_01_init(net: nn.Module):
    """initialize tensor with torch.randn and scale by 0.1"""
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                device = module.weight.device
                in_channels, _, kernel_size1, kernel_size2 = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / \
                    np.sqrt(kernel_size1 * kernel_size2 * in_channels) * 0.1
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                device = module.weight.device
                in_channels, _ = module.weight.shape
                module.weight[:] = torch.randn(module.weight.shape, device=device) / np.sqrt(in_channels) * 0.1
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                continue

    return net


def mixup(input_, target, alpha=0.2):
    """Returns mixed inputs and target"""
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input_.size(0))
    perm_input = input_[perm]
    perm_target = target[perm]
    return input_.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def one_hot(label, num_classes, smoothing_eps=None):
    """use smoothing_eps to smooth y"""
    if smoothing_eps is None:
        one_hot_y = F.one_hot(label, num_classes).float()
        return one_hot_y
    one_hot_y = F.one_hot(label, num_classes).float()
    value1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
    value0 = smoothing_eps / float(num_classes)
    new_y = one_hot_y * (value1 - value0) + value0
    return new_y


def cross_entropy(logit, target):
    """"compute cross entropy loss"""
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def config_dist_env_and_opt(opt):
    """initialize for different training strategies"""
    opt = copy.copy(opt)

    # set world_size, gpu, global rank
    if opt.dist_mode == 'cpu':
        opt.gpu = None
        opt.world_size = 1
        opt.rank = 0
    elif opt.dist_mode == 'single':
        if opt.gpu is None:
            opt.gpu = 0
        opt.world_size = 1
        opt.rank = 0
        torch.cuda.set_device(opt.gpu)
    elif opt.dist_mode == 'horovod':
        hvd.init()
        # Horovod: pin GPU to local rank.
        opt.gpu = hvd.local_rank()
        torch.cuda.set_device(opt.gpu)
        opt.world_size = hvd.size()
        opt.rank = hvd.rank()
    elif opt.dist_mode == 'apex':
        opt.gpu = 0
        opt.world_size = 1
        opt.rank = 0

        if 'WORLD_SIZE' in os.environ:
            opt.distributed = int(os.environ['WORLD_SIZE']) > 1

        if opt.distributed:
            opt.gpu = opt.local_rank
            torch.cuda.set_device(opt.gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            opt.world_size = torch.distributed.get_world_size()
            opt.rank = torch.distributed.get_rank()
    else:
        raise ValueError(f'unknown dist_mode={opt.dist_mode}')

    if not opt.dist_mode == 'cpu':
        torch.backends.cudnn.benchmark = True

    # adjust batch_size and learning rate
    if opt.batch_size is None:
        opt.batch_size = opt.batch_size_per_gpu * opt.world_size
    if opt.lr is None:
        opt.lr = opt.lr_per_256 * opt.batch_size / 256.0
    if opt.target_lr is None:
        opt.target_lr = opt.target_lr_per_256 * opt.batch_size / 256.0

    return opt


def init_model(model, opt, argv):  # pylint: disable=W0613
    """select the network initialization method"""
    if hasattr(opt, 'weight_init') and opt.weight_init == 'xavier':
        network_weight_xavier_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'MSRAPrelu':
        network_weight_MSRAPrelu_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'stupid':
        network_weight_stupid_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'zero':
        network_weight_zero_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == '01':
        network_weight_01_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'custom':
        if not hasattr(model, 'init_parameters'):
            my_logger.info('Warning! No init_parameters found')
        else:
            model.init_parameters()
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'None':
        my_logger.info('Warning!!! model loaded without initialization !')
    else:
        raise ValueError('Unknown weight_init')

    if hasattr(opt, 'bn_momentum') and opt.bn_momentum is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = opt.bn_momentum

    if hasattr(opt, 'bn_eps') and opt.bn_eps is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = opt.bn_eps

    return model


def get_optimizer(model, opt):
    """configure model optimizer"""
    params = split_weights(model)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=opt.nesterov)
    elif opt.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(params,
                                         opt.lr,
                                         opt.adadelta_rho,
                                         opt.adadelta_eps,
                                         weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, opt.lr, weight_decay=opt.weight_decay)

    elif opt.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, opt.lr, alpha=0.9, momentum=opt.momentum, weight_decay=opt.weight_decay)

    else:
        raise ValueError('Unknown optimizer: ' + opt.optimizer)

    return optimizer


def load_model(model, load_parameters_from, strict_load=False, map_location='cpu'):
    """load model state dict from load_parameters_from

        :param model: model
        :param load_parameters_from: checkpoint file
        :return: model
    """
    my_logger.info('loading params from %s', load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    save_model = model.module if isinstance(model, DDP) else model
    save_model.load_state_dict(state_dict, strict=strict_load)

    return model


def resume_checkpoint(model, optimizer, checkpoint_filename, opt, map_location='cpu'):
    """restore the model and optimizer

        :param: model: model
        :param: optimizer: optimizer
        :param: checkpoint_filename: checkpoint file
        :return model, optimizer
    """
    my_logger.info('resuming from %s', checkpoint_filename)
    checkpoint = torch.load(checkpoint_filename, map_location=map_location)
    assert 'state_dict' in checkpoint
    state_dict = checkpoint['state_dict']

    save_model = model.module if isinstance(model, DDP) else model
    save_model.load_state_dict(state_dict, strict=True)
    # model.load_state_dict(state_dict, strict=True)

    optimizer.load_state_dict(checkpoint['optimizer'])

    opt.start_epoch = checkpoint['epoch'] + 1
    training_status_info = checkpoint['training_status_info']

    return model, optimizer, training_status_info, opt


def config_model_optimizer_hvd_and_apex(model, optimizer, opt):
    """configure horovod and apex"""
    if opt.dist_mode == 'horovod' and not opt.independent_training:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if opt.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             compression=compression,
                                             backward_passes_per_step=opt.batches_per_allreduce)
        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if opt.apex:
        if opt.apex_loss_scale != 'dynamic':
            apex_loss_scale = float(opt.apex_loss_scale)
        else:
            apex_loss_scale = 'dynamic'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.apex_opt_level, loss_scale=apex_loss_scale)

    return model, optimizer


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt, num_train_samples, no_acc_eval=False):
    """ model training

        :param train_loader: train dataset loader
        :param model: model
        :param criterion: loss criterion
        :param optimizer:
        :param epoch: current epoch
        :param num_train_samples: total number of samples in train_loader
        :param no_acc_eval (bool): accuray eval in model training
        :return:
    """

    info = {}

    losses = AverageMeter('Loss ', ':6.4g')
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')
    # switch to train mode
    model.train()

    lr_scheduler = global_utils.LearningRateScheduler(mode=opt.lr_mode,
                                                      lr=opt.lr,
                                                      num_training_instances=num_train_samples,
                                                      target_lr=opt.target_lr,
                                                      stop_epoch=opt.epochs,
                                                      warmup_epoch=opt.warmup,
                                                      stage_list=opt.lr_stage_list,
                                                      stage_decay=opt.lr_stage_decay)
    lr_scheduler.update_lr(batch_size=epoch * num_train_samples)

    optimizer.zero_grad()
    batches_per_allreduce_count = 0

    for i, (input_, target) in enumerate(train_loader):

        if not opt.independent_training:
            lr_scheduler.update_lr(batch_size=input_.shape[0] * opt.world_size)
        else:
            lr_scheduler.update_lr(batch_size=input_.shape[0])
        current_lr = lr_scheduler.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * opt.batches_per_allreduce

        bool_label_smoothing = False
        bool_mixup = False

        if not opt.dist_mode == 'cpu':
            input_ = input_.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

        transformed_target = target

        with torch.no_grad():
            if hasattr(opt, 'label_smoothing') and opt.label_smoothing:
                bool_label_smoothing = True
            if hasattr(opt, 'mixup') and opt.mixup:
                bool_mixup = True

            if bool_label_smoothing and not bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes, smoothing_eps=0.1)

            if not bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes)
                input_, transformed_target = mixup(input_, transformed_target)

            if bool_label_smoothing and bool_mixup:
                transformed_target = one_hot(target, num_classes=opt.num_classes, smoothing_eps=0.1)
                input_, transformed_target = mixup(input_, transformed_target)

        # compute output

        output = model(input_)
        model_saved = model.module if hasattr(model, 'module') else model
        logit_loss = criterion(output, transformed_target)
        ts_feature_loss, ts_logit_loss = model_saved.compute_ts_distill_loss()
        loss = logit_loss + opt.teacher_feature_weight * ts_feature_loss + opt.teacher_logit_weight * ts_logit_loss

        # measure accuracy and record loss
        input_size = int(input_.size(0))
        if not no_acc_eval:
            # pylint: disable=unbalanced-tuple-unpacking
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(float(acc1[0]), input_size)
            top5.update(float(acc5[0]), input_size)
        else:
            acc1 = [0]
            acc5 = [0]

        losses.update(float(loss), input_size)

        if opt.apex:
            if opt.dist_mode == 'horovod':

                if batches_per_allreduce_count >= opt.batches_per_allreduce:
                    optimizer.zero_grad()
                    batches_per_allreduce_count = 0

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    batches_per_allreduce_count += 1

                    if opt.grad_clip is not None:
                        torch.nn.utils.clip_grad_value_(model_saved.parameters(), opt.grad_clip)

                if batches_per_allreduce_count >= opt.batches_per_allreduce:
                    optimizer.synchronize()
                    with optimizer.skip_synchronize():
                        optimizer.step()
            else:
                # if batches_per_allreduce_count >= opt.batches_per_allreduce:
                optimizer.zero_grad()
                batches_per_allreduce_count = 0

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    if opt.grad_clip is not None:
                        torch.nn.utils.clip_grad_value_(model_saved.parameters(), opt.grad_clip)

                # if batches_per_allreduce_count >= opt.batches_per_allreduce:
                optimizer.step()
        else:
            if batches_per_allreduce_count >= opt.batches_per_allreduce:
                optimizer.zero_grad()
                batches_per_allreduce_count = 0

            loss.backward()
            batches_per_allreduce_count += 1

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(model_saved.parameters(), opt.grad_clip)

            if batches_per_allreduce_count >= opt.batches_per_allreduce:
                optimizer.step()

        if i % opt.print_freq == 0:

            print(
                f'<rank {opt.rank}> Train epoch={epoch}, i={i}, loss={float(loss):4g}, \
                  logit_loss={float(logit_loss):4g}, ts_feature_loss={float(ts_feature_loss):4g}, \
                  ts_logit_loss={float(ts_logit_loss):4g}, \
                  acc1={float(acc1[0]):4g}%, acc5={float(acc5[0]):4g}%, lr={current_lr:4g}')

    top1_acc_avg = top1.avg
    top5_acc_avg = top5.avg
    losses_acc_avg = losses.avg

    # if distributed, sync
    if opt.dist_mode == 'horovod' and (not opt.independent_training):
        sync_tensor = torch.tensor([top1.sum, top1.count, top5.sum, top5.count,
                                    losses.sum, losses.count], dtype=torch.float32)
        hvd.allreduce(sync_tensor, name='sync_tensor_topk_acc')
        top1_acc_avg = (sync_tensor[0] / sync_tensor[1]).item()
        top5_acc_avg = (sync_tensor[2] / sync_tensor[3]).item()
        losses_acc_avg = (sync_tensor[4] / sync_tensor[5]).item()
    elif opt.dist_mode == 'apex' and opt.distributed:
        sync_tensor = torch.tensor([top1.sum, top1.count, top5.sum, top5.count,
                                    losses.sum, losses.count], dtype=torch.float32).cuda()
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        top1_acc_avg = (sync_tensor[0] / sync_tensor[1]).item()
        top5_acc_avg = (sync_tensor[2] / sync_tensor[3]).item()
        losses_acc_avg = (sync_tensor[4] / sync_tensor[5]).item()
    else:
        pass

    info['losses_acc'] = losses_acc_avg
    info['top1_acc'] = top1_acc_avg
    info['top5_acc'] = top5_acc_avg

    return info


def validate(val_loader, model, criterion, opt, epoch='N/A'):
    """ model evaluation

        :param val_loader: validate dataset loader
        :param model: model
        :param criterion: loss criterion
        :return:
    """

    losses = AverageMeter('Loss ', ':6.4g')
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')

    # switch to evaluate mode
    model.eval()

    epoch = opt.start_epoch if epoch == 'N/A' else epoch

    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):

            transformed_target = target
            if (hasattr(opt, 'label_smoothing') and opt.label_smoothing) or (hasattr(opt, 'mixup') and opt.mixup):
                transformed_target = one_hot(transformed_target, num_classes=opt.num_classes, smoothing_eps=None)

            if not opt.dist_mode == 'cpu':
                input_ = input_.cuda(opt.gpu, non_blocking=True)
                target = target.cuda(opt.gpu, non_blocking=True)
                transformed_target = transformed_target.cuda(opt.gpu, non_blocking=True)

            # compute output
            output = model(input_)
            if criterion is not None:
                loss = criterion(output, transformed_target)
            else:
                loss = torch.tensor([0])

            # pylint: disable=unbalanced-tuple-unpacking
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            input_size = int(input_.size(0))

            losses.update(float(loss), input_size)
            top1.update(float(acc1[0]), input_size)
            top5.update(float(acc5[0]), input_size)

            if i % opt.print_freq == 0:
                print(f'<rank {opt.rank}> Eval epoch={epoch}, i={i},\
                        loss={float(loss):4g}, acc1={float(acc1[0]):4g}%, acc5={float(acc5[0]):4g}%')

    top1_acc_avg = top1.avg
    top5_acc_avg = top5.avg
    losses_acc_avg = losses.avg
    total_val_count = top1.count

    # if distributed, sync
    if opt.dist_mode == 'horovod' and (not opt.independent_training):
        sync_tensor = torch.tensor([top1.sum, top1.count, top5.sum, top5.count,
                                    losses.sum, losses.count], dtype=torch.float32)
        sync_tensor = hvd.allreduce(sync_tensor, average=False, name='sync_tensor_topk_acc')
        top1_acc_avg = (sync_tensor[0] / sync_tensor[1]).item()
        top5_acc_avg = (sync_tensor[2] / sync_tensor[3]).item()
        losses_acc_avg = (sync_tensor[4] / sync_tensor[5]).item()
        total_val_count = sync_tensor[1].item()
    elif opt.dist_mode == 'apex' and opt.distributed:
        sync_tensor = torch.tensor([top1.sum, top1.count, top5.sum, top5.count,
                                    losses.sum, losses.count], dtype=torch.float32).cuda()
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        top1_acc_avg = (sync_tensor[0] / sync_tensor[1]).item()
        top5_acc_avg = (sync_tensor[2] / sync_tensor[3]).item()
        losses_acc_avg = (sync_tensor[4] / sync_tensor[5]).item()
        total_val_count = sync_tensor[1].item()
    else:
        pass

    my_logger.info(' * Validate Acc@1 %.3f Acc@5 %.3f, n_val=%d', top1_acc_avg, top5_acc_avg, total_val_count)
    # if opt.rank == 0:
    #     print(' * Validate Acc@1 {:.3f} Acc@5 {:.3f}, n_val={}'.format(top1_acc_avg, top5_acc_avg, total_val_count))

    return {'top1_acc': top1_acc_avg, 'top5_acc': top5_acc_avg, 'losses_acc': losses_acc_avg}


def train_all_epochs(opt, model, optimizer, train_sampler, train_loader, criterion, val_loader, num_train_samples=None,
                     no_acc_eval=False, save_all_ranks=False, training_status_info=None, save_params=True, writer=None):
    """model training and evaluation, and saving best and latest model
        :param opt: training and evaluation configure
        :param model: model
        :param optimizer: optimizer
        :param train_sampler: train dataset sampler
        :param train_loader: train dataset loader
        :param criterion: loss criterion
        :param val_loader: val dataset loader
        :param no_acc_eval (bool): accuray eval in model training
        :param num_train_samples: total number of samples in train_loader
        :param save_all_ranks (bool): default False
        :param training_status_info (dict): record training status
        :param save_params (bool): save model parameters
        :param writer: tensorboard
        :return:
    """

    timer_start = time.time()

    if training_status_info is None:
        training_status_info = {}
        training_status_info['best_acc1'] = 0
        training_status_info['best_acc5'] = 0
        training_status_info['best_acc1_at_epoch'] = 0
        training_status_info['best_acc5_at_epoch'] = 0
        training_status_info['training_elasped_time'] = 0
        training_status_info['validation_elasped_time'] = 0

    if num_train_samples is None:
        num_train_samples = len(train_loader)

    for epoch in range(opt.start_epoch, opt.epochs):
        my_logger.info('--- Start training epoch %d', epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        training_timer_start = time.time()
        train_one_epoch_info = train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt, num_train_samples,
                                               no_acc_eval=no_acc_eval)

        training_status_info['training_elasped_time'] += time.time() - training_timer_start
        model_saved = model.module if hasattr(model, 'module') else model

        # evaluate on validation set
        if val_loader is not None:
            validation_timer_start = time.time()
            validate_info = validate(val_loader, model, criterion, opt, epoch=epoch)
            training_status_info['validation_elasped_time'] += time.time() - validation_timer_start
            acc1 = validate_info['top1_acc']
            acc5 = validate_info['top5_acc']
        else:
            acc1 = 0
            acc5 = 0

        # remember best acc@1 and save checkpoint
        is_best_acc1 = acc1 > training_status_info['best_acc1']
        is_best_acc5 = acc5 > training_status_info['best_acc5']
        training_status_info['best_acc1'] = max(acc1, training_status_info['best_acc1'])
        training_status_info['best_acc5'] = max(acc5, training_status_info['best_acc5'])

        if is_best_acc1:
            training_status_info['best_acc1_at_epoch'] = epoch
        if is_best_acc5:
            training_status_info['best_acc5_at_epoch'] = epoch

        elasped_hour = (time.time() - timer_start) / 3600
        remaining_hour = (time.time() - timer_start) / float(epoch - opt.start_epoch + 1) * (opt.epochs - epoch) / 3600
        num_samples = num_train_samples * (epoch + 1)
        if opt.rank == 0:
            print(
                f"--- Epoch={epoch}, Elasped hour={elasped_hour}, Remaining hour={remaining_hour},\
                  Training Speed={num_samples / float(training_status_info['training_elasped_time'] + 1e-8):4g},\
                  best_acc1={training_status_info['best_acc1']:4g},\
                  best_acc1_at_epoch={training_status_info['best_acc1_at_epoch']},\
                  best_acc5={training_status_info['best_acc5']},\
                  best_acc5_at_epoch={training_status_info['best_acc5_at_epoch']}")

        if opt.rank == 0 and writer is not None:
            writer.add_scalar('Train/Loss', train_one_epoch_info['losses_acc'], epoch)
            writer.add_scalar('Train/Top-1 Acc', train_one_epoch_info['top1_acc'], epoch)
            writer.add_scalar('Train/Top-5 Acc', train_one_epoch_info['top5_acc'], epoch)

            writer.add_scalar('Test/Loss', validate_info['losses_acc'], epoch)
            writer.add_scalar('Test/Top-1 Acc', validate_info['top1_acc'], epoch)
            writer.add_scalar('Test/Top-5 Acc', validate_info['top5_acc'], epoch)

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.close()

        # ----- save latest epoch -----#
        if save_params and (opt.rank == 0 or save_all_ranks) and \
                ((epoch + 1) % opt.save_freq == 0 or epoch + 1 == opt.epochs):
            checkpoint_filename = os.path.join(opt.save_dir, f'latest-params_rank{opt.rank}.pth')
            save_checkpoint(checkpoint_filename, {
                'epoch': epoch,
                'state_dict': model_saved.state_dict(),
                'optimizer': optimizer.state_dict(),
                'top1_acc': acc1,
                'top5_acc': acc5,
                'training_status_info': training_status_info
            })

        # ----- save best parameters -----#
        if save_params and is_best_acc1 and (opt.rank == 0 or save_all_ranks):
            checkpoint_filename = os.path.join(opt.save_dir, f'best-params_rank{opt.rank}.pth')
            save_checkpoint(checkpoint_filename, {
                'epoch': epoch,
                'state_dict': model_saved.state_dict(),
                'top1_acc': acc1,
                'top5_acc': acc5,
                'training_status_info': training_status_info
            })

        # ----- save best parameters for student model -----#
        if save_params and is_best_acc1 and (opt.rank == 0 or save_all_ranks):
            checkpoint_filename = os.path.join(opt.save_dir, f'student_best-params_rank{opt.rank}.pth')
            save_checkpoint(checkpoint_filename, {
                'epoch': epoch,
                'state_dict': model_saved.student_model.state_dict(),
                'top1_acc': acc1,
                'top5_acc': acc5,
                'training_status_info': training_status_info
            })

    return training_status_info


def main(opt, argv):
    """Train and test

        :param argv: sys.argv
        :param logger: logging.getLogger
        :param writer: tensorboard
        :return:
    """

    assert opt.save_dir is not None

    job_done_fn = os.path.join(opt.save_dir, 'train_image_classification.done')
    if os.path.isfile(job_done_fn):
        print('skip ' + job_done_fn)
        return
    opt.distributed = False
    opt = config_dist_env_and_opt(opt)

    if opt.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    writer = None
    if opt.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs -port", view at http://localhost:6006/')
        writer = SummaryWriter()

    global my_logger  # pylint: disable=global-statement,invalid-name
    # create log
    if opt.rank == 0:
        log_filename = os.path.join(opt.save_dir, 'train_image_classification.log')
        # global_utils.create_logging(log_filename=log_filename)
        my_logger = get_logger(log_filename=log_filename)
    else:
        # global_utils.create_logging(log_filename=None, level=logging.ERROR)
        my_logger = get_logger(log_filename=None, level=logging.ERROR)

    my_logger.info('argv=\n%s', str(argv))
    my_logger.info('opt=\n%s', str(opt))
    my_logger.info('-----')

    # load dataset
    tmp_opt = copy.copy(opt)
    tmp_opt.input_image_size = max(opt.input_image_size, opt.teacher_input_image_size)
    data_loader_info = DataLoader.get_data(tmp_opt, argv)
    train_loader = data_loader_info['train_loader']
    val_loader = data_loader_info['val_loader']
    train_sampler = data_loader_info['train_sampler']
    num_train_samples = DataLoader.params_dict[opt.dataset]['num_train_samples']

    # create model
    student_model = ModelLoader.get_model(opt, argv)
    student_model = init_model(student_model, opt, argv)
    my_logger.info('loading student_model:')
    my_logger.info(str(student_model))

    # create teacher model
    assert opt.teacher_pretrained or opt.teacher_load_parameters_from is not None

    teacher_opt = copy.copy(opt)
    # rename all cmd options start with 'teacher_' to be standard options
    tmp_teacher_dict = {}
    for k, value in opt.__dict__.items():
        if k.startswith('teacher_'):
            new_key = k[len('teacher_'):]
            tmp_teacher_dict[new_key] = value

    teacher_opt.__dict__.update(tmp_teacher_dict)
    teacher_model = ModelLoader.get_model(teacher_opt, argv)
    my_logger.info('loading teacher_model:')
    my_logger.info(str(teacher_model))

    my_logger.info('create teacher-student model')
    # create teacher-student model
    model = TeacherStudentDistillNet(teacher_model=teacher_model, student_model=student_model,
                                     opt=opt, argv=argv, target_downsample_ratio=opt.target_downsample_ratio)

    # get optimizer
    optimizer = get_optimizer(model, opt)

    my_logger.info('optimizer is :')
    my_logger.info(str(optimizer))

    if opt.sync_bn:
        my_logger("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    # set device
    if opt.gpu is not None:
        # torch.cuda.set_device(opt.gpu)
        model.cuda().to(memory_format=memory_format)
        print('rank={%d}, using GPU {%d}', opt.rank, opt.gpu)

    # define loss function (criterion)
    if (hasattr(opt, 'label_smoothing') and opt.label_smoothing) or (hasattr(opt, 'mixup') and opt.mixup):
        criterion = cross_entropy
    else:
        criterion = nn.CrossEntropyLoss()
        if not opt.dist_mode == 'cpu':
            criterion = criterion.cuda(opt.gpu)

    # hvd and apex
    model, optimizer = config_model_optimizer_hvd_and_apex(model, optimizer, opt)

    if opt.distributed:
        model = DDP(model, delay_allreduce=True)

    training_status_info = {}
    training_status_info['best_acc1'] = 0
    training_status_info['best_acc5'] = 0
    training_status_info['best_acc1_at_epoch'] = 0
    training_status_info['best_acc5_at_epoch'] = 0
    training_status_info['training_elasped_time'] = 0
    training_status_info['validation_elasped_time'] = 0

    map_location = 'cpu'
    if opt.gpu is not None:
        map_location = f'cuda:{opt.gpu}'

    if opt.auto_resume and opt.resume is None:
        latest_pth_fn = os.path.join(opt.save_dir, 'latest-params_rank0.pth')
        if os.path.isfile(latest_pth_fn):
            logging.info(('auto-resume from %s', latest_pth_fn))
            model, optimizer, training_status_info, opt = resume_checkpoint(model, optimizer, latest_pth_fn, opt,
                                                                            map_location=map_location)

    if opt.resume:
        assert not opt.auto_resume
        logging.info(('resume from %s', opt.resume))
        model, optimizer, training_status_info, opt = resume_checkpoint(model, optimizer, opt.resume, opt,
                                                                        map_location=map_location)

    if not opt.evaluate_only:
        training_status_info = train_all_epochs(
            opt, model, optimizer, train_sampler, train_loader, criterion, val_loader,
            num_train_samples=num_train_samples, no_acc_eval=False, save_all_ranks=False,
            training_status_info=training_status_info, writer=writer)
    else:
        validate(val_loader, model, criterion, opt)

    # mark job done
    global_utils.save_pyobj(job_done_fn, training_status_info)


if __name__ == "__main__":
    option = global_utils.parse_cmd_options(sys.argv)
    defaut_log_filename = os.path.join(option.save_dir, 'train_image_classification.log')
    my_logger = get_logger(log_filename=defaut_log_filename)
    main(option, sys.argv)
