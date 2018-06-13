#!/bin/bash

#MODEL=resnet_v2_18_slim
MODEL=resnet_v2_18     # reduce --batch if OOM
#MODEL=resnet_v2_50

../train-basic-keypoints.py \
                     --picpac_dump 0 \
                     --channels 1 \
                     --db owls/forarm1/db \
                     --backbone $MODEL \
                     --backbone_stride 16 \
                     --feature_channels 64 \
                     --batch 8 \
                     --fix_width 512 \
                     --fix_height 512 \
                     --max_size 512 \
                     --classes 1 \
                     --val_epochs 10  \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-5 \
                     --max_epochs 400 \
                     --model $MODEL \
                     --offset_weight 0.1 \
                     $*

#../predict-basic-keypoints.py --channels 1 --input_db val.db $*
