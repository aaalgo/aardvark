#!/bin/bash

MODEL=resnet_v2_18_slim
#MODEL=resnet_v2_18     # reduce --batch if OOM
#MODEL=resnet_v2_50

../train-fcn-slim.py \
                     --db train.db \
                     --val_db val.db \
                     --backbone $MODEL \
                     --backbone_stride 16 \
                     --reduction 2 \            # reduce channels by this factor before deconvolution
                     --batch 8 \
                     --fix_width 640 \
                     --fix_height 400 \
                     --max_size 640 \
                     --classes 2 \
                     --val_epochs 1  \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4 \
                     --dice \
                     --model $MODEL \           # save to this path
                     $*

