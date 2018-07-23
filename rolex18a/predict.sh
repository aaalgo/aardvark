#!/bin/bash

../predict-faster-rcnn.py --input_db rolexlogotest1.db  --max_size 1000 --min_size 800 --vis_downsize 4 --anchor_th 0.95 $* 
