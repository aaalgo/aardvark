#!/usr/bin/env python3
import os
import math
import sys
# C++ code, python3 setup.py build
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
import cv2
from faster_rcnn import FasterRCNN
import cpp

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_integer('backbone_stride', 16, '')


class Model (FasterRCNN):
    def __init__ (self):
        super().__init__(FLAGS.backbone_stride)
        pass

    def build_backbone (self, images):
        self.backbone = aardvark.create_stock_slim_network(FLAGS.backbone, images, self.is_training, global_pool=False, stride=FLAGS.backbone_stride, scope='bb1')
        self.backbone_stride = FLAGS.backbone_stride
        pass

    def rpn_activation (self, channels, stride):
        upscale = self.backbone_stride // stride
        with slim.arg_scope(aardvark.default_argscope(self.is_training)):
            return slim.conv2d_transpose(self.backbone, channels, 2*upscale, upscale, activation_fn=None)
        pass

    def rpn_parameters (self, channels, stride):
        upscale = self.backbone_stride // stride
        with slim.arg_scope(aardvark.default_argscope(self.is_training)):
            return slim.conv2d_transpose(self.backbone, channels, 2*upscale, upscale, activation_fn=None)
        pass

def main (_):
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

