#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

save_dir=../../save_dir/Zen_NAS_ImageNet_latency0.1ms
mkdir -p ${save_dir}



resolution=224
budget_latency=1e-4
max_layers=10
population_size=512
epochs=480
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for sufficient searching






horovodrun -np 8 -H localhost:8 python ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
  --dist_mode horovod --workers_per_gpu 8 --sync_bn \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/zennet_imagenet1k_latency01ms_res224.txt \
  --teacher_arch geffnet_tf_efficientnet_b3_ns \
  --teacher_pretrained \
  --teacher_input_image_size 320 \
  --teacher_feature_weight 1.0 \
  --teacher_logit_weight 1.0 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --target_downsample_ratio 16 \
  --batch_size_per_gpu 64 --save_dir ${save_dir}/ts_effnet_b3ns_epochs${epochs}