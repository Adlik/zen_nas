#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

save_dir=../../save_dir/searchv4_Zen_NAS_ImageNet_latency0.1ms
mkdir -p ${save_dir}


resolution=224
budget_latency=1e-4
max_layers=10
population_size=512
epochs=480
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for sufficient searching

echo "SuperConvK3BNRELU(3,32,2,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,128,2,64,1)\
SuperResK3K3(128,256,2,128,1)SuperResK3K3(256,512,2,256,1)\
SuperConvK1BNRELU(256,512,1,1)" > ${save_dir}/init_plainnet.txt



python -m torch.distributed.launch --nproc_per_node=8 ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
  --dist_mode apex --workers_per_gpu 6 --apex \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/searchv4_imagenet1k_latency01ms_ress224.txt \
  --teacher_arch geffnet_tf_efficientnet_b3_ns \
  --teacher_pretrained \
  --teacher_input_image_size 320 \
  --teacher_feature_weight 1.0 \
  --teacher_logit_weight 1.0 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --target_downsample_ratio 16 \
  --batch_size_per_gpu 256 --save_dir ${save_dir}/ts_effnet_b3ns_epochs${epochs}