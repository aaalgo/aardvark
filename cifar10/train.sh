#!/bin/bash

./download.sh http://www.aaalgo.com/picpac/datasets/cifar/cifar10-train.picpac
./download.sh http://www.aaalgo.com/picpac/datasets/cifar/cifar10-test.picpac

../train-cls-slim-official.py \
                     --db cifar10-train.picpac \
                     --val_db cifar10-test.picpac \
                     --net resnet_v2_18_cifar \
                     --batch 128 \
                     --fix_width 32 \
                     --fix_height 32 \
                     --clip_shift 4 \
                     --clip_stride 0 \
                     --classes 10 \
                     --val_epochs 1  \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4
