#!/bin/bash

../train-cls-slim.py \
                     --nocache \
                     --db scratch/train.db \
                     --net resnet_v2_18 \
                     --batch 32 \
                     --max_size 400 \
                     --fix_width 200 \
                     --fix_height 112 \
                     --classes 3 \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4 \
                     --shift 8 \
                     --channels 3 \
                     $*
