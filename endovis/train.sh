#!/bin/bash

../train-cls-slim.py \
                     --nocache \
                     --db scratch/train.db \
                     --val_db scratch/val.db \
                     --net resnet_v2_18 \
                     --batch 32 \
                     --fix_width 400 \
                     --fix_height 400 \
                     --classes 14 \
                     --patch_slim \
                     --weight_decay 2.5e-4 \
                     --channels 3 \
                     $*
