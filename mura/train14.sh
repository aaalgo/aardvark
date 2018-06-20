#!/bin/bash

../train-cls-slim.py \
                     --nocache \
                     --db scratch/train.db \
                     --val_db scratch/val0.db \
                     --val_epochs 1 \
                     --net resnet_v2_18 \
                     --batch 32 \
                     --max_size 400 \
                     --fix_width 400 \
                     --fix_height 400 \
                     --classes 14 \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4 \
                     --shift 16 \
                     --channels 1 \
                     $*
