#!/bin/bash

MODEL=resnet_v2_18 

../train-faster-rcnn.py \
                     --compact \
                     --nocache \
                     --db train.db \
                     --backbone $MODEL \
                     --backbone_stride 16 \
                     --batch 32 \
                     --fix_width 400 \
                     --fix_height 400 \
                     --max_size 400 \
                     --classes 2 \
                     --patch_slim \
                     --weight_decay 2.5e-4 \
                     --model rcnn_res18 \
                     --lower_th 0.05 \
                     --upper_th 0.25 \
		             --anchor_th 0.5 \
                     --match_th 0.3 \
                     --nms_max 256 \
                     --rpnonly \
                     --anchor_stride 4 \
                     --max_epochs 2000 \
                     --augments augments.json \
                     --priors priors \
                     $*


# predict
#../predict-fcn.py --max_size 640 --model resnet_v2_50/30 --db val.db

