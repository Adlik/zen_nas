'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
process images andload imagenet and cifar
'''
# pylint: disable=W0613,not-callable,invalid-name,too-many-locals,too-many-arguments
import os
import sys
import math
import torch
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torchvision import datasets
import numpy as np
import PIL
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import hotfix.transforms
    import autoaugment
except ImportError:
    print('fail to import zen_nas modules')

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

LIGHTING_PARAM = 0.1

params_dict = {
    'imagenet': {
        'train_dir': os.path.expanduser('~/data/imagenet/images/train/'),
        'val_dir': os.path.expanduser('~/data/imagenet/images/val/'),
        'num_train_samples': 1281167,
        'num_val_samples': 50000,
        'num_classes': 1000,
    },
    'myimagenet100': {
        'train_dir': os.path.expanduser('~/data/myimagenet100/train/'),
        'val_dir': os.path.expanduser('~/data/myimagenet100/val/'),
        'num_train_samples': 129395,
        'num_val_samples': 5000,
        'num_classes': 100,
    },
    'cifar10': {
        'train_dir': os.path.expanduser('~/data/pytorch_cifar10'),
        'val_dir': os.path.expanduser('~/data/pytorch_cifar10'),
        'num_train_samples': 50000,
        'num_val_samples': 10000,
        'num_classes': 10,
    },

    'cifar100': {
        'train_dir': os.path.expanduser('~/data/pytorch_cifar100'),
        'val_dir': os.path.expanduser('~/data/pytorch_cifar100'),
        'num_train_samples': 50000,
        'num_val_samples': 10000,
        'num_classes': 100,
    },
}


def fast_collate(batch, memory_format):
    """conver array to tensor"""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    # print('image size: ',imgs[0].size())
    height = imgs[0].size()[1]
    weight = imgs[0].size()[2]
    tensor = torch.zeros((len(imgs), 3, height, weight), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        # nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


# pylint: disable=too-few-public-methods
class Lighting():
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def load_imagenet_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_image_folder_class,
                       dataloader_testing, channel_last=False):
    """load imagenet dataset

        :param dataset_name (str): dataset name
        :param set_name (str): train or val
        :param batch_size (int): batch size
        :param train_augment (bool): data augmentation
        :param random_erase (bool): random erase
        :param auto_augment (bool): Auto Augmentation
        :param input_image_size (int): input image size
        :param input_image_crop (float): input image crop ratio
        :param rank (int): rank
        :param world_size (int): number of GPUs
        :param shuffle (bool): shuffle
        :param num_workers (int): the number of workers
        :param drop_last (bool): drop last
        :param dataset_image_folder_class: datasets.ImageFolder
        :param dataloader_testing (bool): dataloader testing
        :param channel_last (bool): channel_last or contiguous_format
        :return data_loader, sampler
    """
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train_augment is False:
        assert random_erase is False and auto_augment is False
        transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC),
                          transforms.CenterCrop(input_image_size),
                          transforms.ToTensor(), transforms_normalize]
    else:
        if auto_augment:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              autoaugment.ImageNetPolicy(),
                              transforms.ToTensor(),
                              Lighting(LIGHTING_PARAM, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        else:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.4, 0.4, 0.4),
                              transforms.ToTensor(),
                              Lighting(LIGHTING_PARAM, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        if random_erase:
            transform_list.append(hotfix.transforms.RandomErasing())

    transformer = transforms.Compose(transform_list)

    the_dataset = dataset_image_folder_class(data_dir, transformer)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        indices_or_sections = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, indices_or_sections)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset)

    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }


# pylint: disable=too-many-branches
def load_cifar_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                    data_dir, input_image_size, input_image_crop, rank, world_size,
                    shuffle, batch_size, num_workers, drop_last, dataset_image_folder_class,
                    dataloader_testing=False):
    """load cifar dataset

        :param dataset_name (str): dataset name
        :param set_name (str): train or val
        :param batch_size (int): batch size
        :param train_augment (bool): data augmentation
        :param random_erase (bool): random erase
        :param auto_augment (bool): Auto Augmentation
        :param input_image_size (int): input image size
        :param input_image_crop (float): input image crop ratio
        :param rank (int): rank
        :param world_size (int): number of GPUs
        :param shuffle (bool): shuffle
        :param num_workers (int): the number of workers
        :param drop_last (bool): drop last
        :param dataset_image_folder_class: datasets.ImageFolder
        :param dataloader_testing (bool): dataloader testing
        :return data_loader, sampler
    """

    transforms_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    if train_augment is False:
        assert random_erase is False and auto_augment is False
        if input_image_size > 32:
            transform_list = [transforms.Resize(input_image_size, interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = []

        transform_list += [transforms.ToTensor(), transforms_normalize]
    else:

        if input_image_size > 32:
            resize_image_size = round(input_image_size / 0.75)
            transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC)]
            transform_list += [transforms.RandomResizedCrop(input_image_size, scale=(0.8, 1.0),
                                                            interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = [transforms.RandomCrop(input_image_size, padding=4)]

        if auto_augment:
            autoaugment_policy = autoaugment.CIFAR10Policy()
            transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                               transforms.ToTensor(),
                               transforms_normalize]
        else:
            transform_list += [transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms_normalize]
        if random_erase:
            transform_list.append(hotfix.transforms.RandomErasing())

    transformer = transforms.Compose(transform_list)

    if dataset_name == 'cifar10':
        the_dataset = datasets.CIFAR10(root=data_dir, train=set_name == 'train', download=True, transform=transformer)
    elif dataset_name == 'cifar100':
        the_dataset = datasets.CIFAR100(root=data_dir, train=set_name == 'train', download=True, transform=transformer)
    else:
        raise ValueError('Unknown dataset_name=' + dataset_name)

    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        indices_or_sections = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, indices_or_sections)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }


# pylint: disable=inconsistent-return-statements
def _get_data_(dataset_name=None, set_name=None, batch_size=None,
               train_augment=False, random_erase=False, auto_augment=False,
               input_image_size=224, input_image_crop=0.875, rank=0, world_size=1,
               shuffle=False, num_workers=6, drop_last=False, dataset_image_folder_class=None,
               dataloader_testing=False, argv=None, channel_last=False):
    """get imagenet/cifar dataset

        :param dataset_name (str): dataset name
        :param set_name (str): train or val
        :param batch_size (int): batch size
        :param train_augment (bool): data augmentation
        :param random_erase (bool): random erase
        :param auto_augment (bool): Auto Augmentation
        :param input_image_size (int): input image size
        :param input_image_crop (float): input image crop ratio
        :param rank (int): rank
        :param world_size (int): number of GPUs
        :param shuffle (bool): shuffle
        :param num_workers (int): the number of workers
        :param drop_last (bool): drop last
        :param dataset_image_folder_class: datasets.ImageFolder
        :param dataloader_testing (bool): dataloader testing
        :param argv: sys.argv
        :param channel_last (bool): channel_last or contiguous_format
        :return data_loader, sampler
    """

    if dataset_name in ['imagenet', 'myimagenet100']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_image_folder_class is None:
            dataset_image_folder_class = datasets.ImageFolder

        return load_imagenet_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                                  random_erase=random_erase, auto_augment=auto_augment,
                                  data_dir=data_dir,
                                  input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                                  world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=drop_last,
                                  dataset_image_folder_class=dataset_image_folder_class,
                                  dataloader_testing=dataloader_testing,
                                  channel_last=channel_last)

    if dataset_name in ['cifar10', 'cifar100']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_image_folder_class is None:
            dataset_image_folder_class = datasets.ImageFolder

        return load_cifar_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                               random_erase=random_erase, auto_augment=auto_augment,
                               data_dir=data_dir,
                               input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                               world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                               num_workers=num_workers, drop_last=drop_last,
                               dataset_image_folder_class=dataset_image_folder_class,
                               dataloader_testing=dataloader_testing)


def get_data(opt, argv):
    """get train/val loader and sampler"""
    dataset_name = opt.dataset
    batch_size = opt.batch_size_per_gpu
    random_erase = opt.random_erase
    auto_augment = opt.auto_augment
    input_image_size = opt.input_image_size
    input_image_crop = opt.input_image_crop
    rank = opt.rank
    world_size = opt.world_size
    num_workers = opt.workers_per_gpu
    channel_last = opt.channels_last

    # check if independent training
    if opt.independent_training:
        rank = 0
        world_size = 1

    # load train set
    set_name = 'train'
    if opt.no_data_augment:
        train_augment = False
    else:
        train_augment = True
    shuffle = True
    drop_last = True
    batch_size *= opt.batches_per_allreduce
    train_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                    input_image_size, input_image_crop, rank, world_size, shuffle,
                                    num_workers, drop_last, dataloader_testing=opt.dataloader_testing,
                                    argv=argv, channel_last=channel_last)

    # load val set
    set_name = 'val'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False

    val_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                  input_image_size, input_image_crop, rank, world_size, shuffle,
                                  num_workers, drop_last, dataloader_testing=opt.dataloader_testing,
                                  argv=argv, channel_last=channel_last)

    return {
        'train_loader': train_dataset_info['data_loader'],
        'val_loader': val_dataset_info['data_loader'],
        'train_sampler': train_dataset_info['sampler'],
        'val_sampler': val_dataset_info['sampler'],
    }
