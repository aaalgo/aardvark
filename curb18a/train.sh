#!/bin/bash

#MODEL=resnet_v2_18_slim
MODEL=resnet_v2_18     # reduce --batch if OOM
#MODEL=resnet_v2_50

reduction=2            # reduce channels by this factor before deconv
                       # resnet has 512 channels and will reduce like (512 ->) 128 -> 64 -> 32 -> 16
../train-fcn-slim.py \
                     --db train.db \
                     --val_db val.db \
                     --backbone $MODEL \
                     --backbone_stride 16 \
                     --reduction $reduction \
                     --batch 8 \
                     --fix_width 640 \
                     --fix_height 400 \
                     --max_size 640 \
                     --classes 2 \
                     --val_epochs 10  \
                     --patch_slim \
                     --noadam \
                     --weight_decay 2.5e-4 \
                     --max_epochs 500 \
                     --dice \
                     --model $MODEL \
                     $*

# predict
# ../predict-fcn.py --max_size 640 --list val.txt --model resnet_v2_18/120
# ../predict-fcn.py --max_size 640 --db val.db --model resnet_v2_18/120
