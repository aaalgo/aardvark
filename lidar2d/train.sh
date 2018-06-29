#!/bin/bash

../train-fcn-slim.py \
                     --db train.db \
                     --net resnet_v2_18 \
                     --batch 16 \
                     --fix_width 400 \
                     --fix_height 400 \
                     --classes 14 \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4 \
                     --shift 16 \
                     --channels 3 \
                     $*
