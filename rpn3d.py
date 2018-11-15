#!/usr/bin/env python3
import os
import math
import sys
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
import cv2
from tf_utils import *
import cpp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('rpn_priors', 'rpn_priors', 'param prior config file')
flags.DEFINE_integer('rpn_params', 3, 'number of parameters per shape')
flags.DEFINE_integer('rpn_stride', 1, 'downsize factor of rpn output')
flags.DEFINE_float('rpn_logits_weight', 1.0, 'loss weight')
flags.DEFINE_float('rpn_params_weight', 1.0, 'loss weight')

class BasicRPN3D:

    def __init__ (self):
        priors = []
        # read in priors
        # what RPN estimates is the delta between priors and the real
        # regression target.
        if os.path.exists(FLAGS.rpn_priors):
            with open(FLAGS.rpn_priors, 'r') as f:
                for l in f:
                    if l[0] == '#':
                        continue
                    vs = [float(v) for v in l.strip().split(' ')]
                    assert len(vs) == FLAGS.rpn_params
                    priors.append(vs)
                    pass
                pass
            pass
        if len(priors) == 0:
            priors.append([1.0] * FLAGS.rpn_params)
            pass
        aardvark.print_red("PRIORS %s" % str(priors))
        self.priors = np.array(priors, dtype=np.float32)
        pass

    def rpn_backbone (self, volume, is_training, stride):
        assert False

    def rpn_logits (self, net, is_training, channels):
        assert False

    def rpn_params (self, net, is_training, channels):
        assert False

    def rpn_generate_shapes (self, shape, anchor_params, priors, n_priors):
        assert False

    def build_rpn (self, volume, is_training, shape=None):
        # volume: input volume tensor
        Z,Y,X = shape
        assert max(Z % FLAGS.rpn_stride, Y % FLAGS.rpn_stride, X % FLAGS.rpn_stride) == 0
        oZ = Z // FLAGS.rpn_stride
        oY = Y // FLAGS.rpn_stride
        oX = X // FLAGS.rpn_stride
        n_priors = self.priors.shape[0]
        n_params = self.priors.shape[1]

        self.gt_anchors = tf.placeholder(tf.float32, shape=(None, oZ, oY, oX, n_priors))
        self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, oZ, oY, oX, n_priors))
        # parameter of that location
        self.gt_params = tf.placeholder(tf.float32, shape=(None, oZ, oY, oX, n_priors,  n_params))
        self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, oZ, oY, oX, n_priors))

        self.backbone = self.rpn_backbone(volume, is_training, FLAGS.rpn_stride)
        logits = self.rpn_logits(self.backbone, is_training, n_priors)
        logits = tf.identity(logits, name='logits')
        self.logits = logits
        self.probs = tf.sigmoid(logits, name='probs')
        params = self.rpn_params(self.backbone, is_training, n_priors * n_params)
        params = tf.identity(params, name='params')
        self.params = params

        # setup losses

        # 1. losses for logits
        logits1 = tf.reshape(logits, (-1,))
        gt_anchors = tf.reshape(self.gt_anchors, (-1,))
        gt_anchors_weight = tf.reshape(self.gt_anchors_weight, (-1,))
        xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits1, labels=tf.cast(gt_anchors, tf.float32))
        xe = tf.reduce_sum(xe * gt_anchors_weight) / (tf.reduce_sum(gt_anchors_weight) + 0.00001)
        xe = tf.identity(xe, name='xe')

        getattr(self, 'metrics', []).append(xe)
        tf.losses.add_loss(xe * FLAGS.rpn_logits_weight)

        # 2. losses for parameters
        priors = tf.constant(self.priors[np.newaxis, :, :], dtype=tf.float32)

        params = tf.reshape(params, (-1, n_priors, n_params))
        gt_params = tf.reshape(self.gt_params, (-1, n_priors, n_params))
        l1 = tf.losses.huber_loss(params, gt_params / priors, reduction=tf.losses.Reduction.NONE, loss_collection=None)
        l1 = tf.reduce_sum(l1, axis=2)
        # l1: ? * n_priors
        l1 = tf.reshape(l1, (-1,))
        gt_params_weight = tf.reshape(self.gt_params_weight, (-1,))

        l1 = tf.reduce_sum(l1 * gt_params_weight) / (tf.reduce_sum(gt_params_weight) + 0.00001)
        l1 = tf.identity(l1, name='l1')

        getattr(self, 'metrics', []).append(l1)
        tf.losses.add_loss(l1 * FLAGS.rpn_params_weight)
        pass


