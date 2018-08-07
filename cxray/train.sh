#!/bin/bash

../train-cls-slim-vis.py \
                     --nocache \
                     --db scratch/train.db \
                     --val_db scratch/val.db \
                     --val_epochs 10 \
                     --net resnet_v2_18 \
                     --batch 16 \
                     --max_size 256 \
                     --fix_width 256 \
                     --fix_height 256 \
                     --classes 14 \
                     --patch_slim \
                     --channels 1 \
                     $*
