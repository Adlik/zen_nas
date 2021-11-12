# zen_nas

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.zen_nas?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/latest?definitionId=5&branchName=main)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/38905)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

Our based code is forked from [ZenNAS](https://github.com/idstcv/ZenNAS).
 We modify the code to make it easier to use, and according to paper
 [Zen-NAS: A Zero-Shot NAS for High-Performance Image Recognition](https://arxiv.org/abs/2102.01063),
  we add some features that ZenNAS does not have.

We mainly made the following changes:

- support Horovod, PyTorch distributed training
- code refactoring for evolutionary search, speed up searching
- repair THCudaTensor sizes too large problem when search latency model

Besides,  as some functions can not work correctly under distributed training, we provide a distributed version.

## Experimental results

We tested the modified code and verified its correctness. The results are as follows:

We used apex with mixed precision to complete the training within 5 days on 8 tesla V100 GPUs,
and the results are consistent with the paper.

|              | paper model accuracy | distributed training accuracy |
| :----------: | :------------------: | :---------------------------: |
| ZenNet-0.1ms |        77.8%         |            77.922%            |
| ZenNet-0.8ms |        83.0%         |            83.214%            |

Note that the same results can be obtained with Horovod, but we took more time to complete the training.
So we recommend using apex for distributed training.

Before the code was released, we reproduced the paper algorithm and searched the model according to the paper's conditions.
The following table shows the comparison results between the search model and the paper model.

|             | paper accuracy | searched model accuracy |
| :---------: | :------------: | :---------------------: |
| latency01ms |     77.8%      |         78.622%         |
| latency05ms |     82.7%      |         82.752%         |
| latency12ms |     83.6%      |         83.466%         |

For proving the effectiveness of the algorithm, we experimented with several different model searches
and get the following result.
We use a single Tesla V100 GPU to evolve the population 50000 times.
|    method     |    model    | search time(hours) | model score |
| :-----------: | :---------: | :----------------: | :---------: |
|    ZenNAS     | latency01ms |      98.4274       |   126.038   |
|        \       | latency05ms |      22.0189       |   243.101   |
|        \       | latency08ms |      28.5952       |   304.323   |
|        \       | latency12ms |      44.6237       |   375.027   |
| modify-ZenNAS | latency01ms |       64.988       |   134.896   |
|        \       | latency05ms |      20.9895       |   245.712   |
|        \       | latency08ms |      25.0358       |   310.629   |
|        \       | latency12ms |       43.239       |   386.669   |

## Reproduce Paper Experiments

### System Requirements

- PyTorch >= 1.6, Python >= 3.7
- By default, ImageNet dataset is stored under \~/data/imagenet;
CIFAR-10/CIFAR-100 is stored under \~/data/pytorch\_cifar10 or \~/data/pytorch\_cifar100
- Pre-trained parameters are cached under \~/.cache/pytorch/checkpoints/zennet\_pretrained

### Package Requirements

- ptflops
- tensorboard >= 1.15 (optional)
- apex

### Pre-trained model download

If you want to evaluate pre-trained models,
please go to [ZenNAS](https://github.com/idstcv/ZenNAS) to download the pre-trained model.

### Evaluate pre-trained models on ImageNet and CIFAR-10/100

To evaluate the pre-trained model on ImageNet using GPU 0:

``` bash
python val.py --fp16 --gpu 0 --arch ${zennet_model_name}
```

where ${zennet\_model\_name} should be replaced by a valid ZenNet model name.
The complete list of model names can be found in the 'Pre-trained Models' section.

To evaluate the pre-trained model on CIFAR-10 or CIFAR-100 using GPU 0:

``` bash
python val_cifar.py --dataset cifar10 --gpu 0 --arch ${zennet_model_name}
```

To create a ZenNet in your python code:

``` python
gpu=0
model = ZenNet.get_ZenNet(opt.arch, pretrained=True)
torch.cuda.set_device(gpu)
torch.backends.cudnn.benchmark = True
model = model.cuda(gpu)
model = model.half()
model.eval()
```

### usage

We supply apex and Horovod distributed training scripts, you can modify other original scripts based on these scripts.
apex script:

```bash
scripts/Zen_NAS_ImageNet_latency0.1ms_train_apex.sh
```

Horovod script:

```bash
scripts/Zen_NAS_ImageNet_latency0.1ms_train.sh
```

If you want to search model, please notice the choices "--fix_initialize" and "--origin".
"--fix_initialize" decides how to initialize population, the algorithm default choice is random initialization.
"--origin" determines how the mutation model is generated.  
When specified "--origin", the mutated model will be produced using the original method.

### Searching on CIFAR-10/100

Searching for CIFAR-10/100 models with budget params < 1M, using different zero-shot proxies:

```bash
scripts/Flops_NAS_cifar_params1M.sh
scripts/GradNorm_NAS_cifar_params1M.sh
scripts/NASWOT_NAS_cifar_params1M.sh
scripts/Params_NAS_cifar_params1M.sh
scripts/Random_NAS_cifar_params1M.sh
scripts/Syncflow_NAS_cifar_params1M.sh
scripts/TE_NAS_cifar_params1M.sh
scripts/Zen_NAS_cifar_params1M.sh
```

### Searching on ImageNet

Searching for ImageNet models, with latency budget on NVIDIA V100 from 0.1 ms/image to 1.2 ms/image at batch size 64 FP16:

```bash
scripts/Zen_NAS_ImageNet_latency0.1ms.sh
scripts/Zen_NAS_ImageNet_latency0.2ms.sh
scripts/Zen_NAS_ImageNet_latency0.3ms.sh
scripts/Zen_NAS_ImageNet_latency0.5ms.sh
scripts/Zen_NAS_ImageNet_latency0.8ms.sh
scripts/Zen_NAS_ImageNet_latency1.2ms.sh
```

Searching for ImageNet models, with FLOPs budget from 400M to 800M:

``` bash
scripts/Zen_NAS_ImageNet_flops400M.sh
scripts/Zen_NAS_ImageNet_flops600M.sh
scripts/Zen_NAS_ImageNet_flops800M.sh
```

## Customize Your Own Search Space and Zero-Shot Proxy

The masternet definition is stored in "Masternet.py".
The masternet takes in a structure string and parses it into a PyTorch nn.Module object.
The structure string defines the layer structure which is implemented in "PlainNet/*.py" files.
For example, in "PlainNet/SuperResK1KXK1.py",
we defined SuperResK1K3K1 block, which consists of multiple layers of ResNet blocks.
To define your block, e.g. ABC_Block, first, implement "PlainNet/ABC_Block.py".
Then in "PlainNet/\_\_init\_\_.py",  after the last line, append the following lines to register the new block definition:

```python
from PlainNet import ABC_Block
_all_netblocks_dict_ = ABC_Block.register_netblocks_dict(_all_netblocks_dict_)
```

After the above registration call, the PlainNet module can parse your customized block from the structure string.

The search space definitions are stored in SearchSpace/*.py. The important function is

```python
gen_search_space(block_list, block_id)
```

block_list is a list of super-blocks parsed by the masternet.
block_id is the index of the block in block_list which will be replaced later by a mutated block
This function must return a list of mutated blocks.

### Direct specify search space

"PlainNet/AABC_Block.py" has defined the candidate blocks,
you can directly specify candidate blocks in the search spaces by passing parameters "--search_space_list".
So you have two methods to specify search spaces.
Taking ResNet-like search space as an example, you can use "--search_space SearchSpace/search_space_XXBL.py" or
"--search_space_list PlainNet/SuperResK1KXK1.py PlainNet/SuperResKXKX.py" to specify search space. Both of them are equivalent.

In scripts, when you choose to use the first method to specify search space,
**you should also add other two parameters "--fix_initialize" and"--origin"**,
so the algorithm will initialize with a fixed model.

The zero-shot proxies are implemented in "ZeroShotProxy/*.py". The evolutionary algorithm is implemented in "evolution_search.py".
"analyze_model.py" prints the FLOPs and model size of the given network.
"benchmark_network_latency.py" measures the network inference latency.
"train_image_classification.py" implements SGD gradient training
and "ts_train_image_classification.py" implements teacher-student distillation.

## Open Source

A few files in this repository are modified from the following open-source implementations:

```text
https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
https://github.com/VITA-Group/TENAS
https://github.com/SamsungLabs/zero-cost-nas
https://github.com/BayesWatch/nas-without-training
https://github.com/rwightman/gen-efficientnet-pytorch
https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
```

## Copyright

Copyright 2021 ZTE corporation. All Rights Reserved.
