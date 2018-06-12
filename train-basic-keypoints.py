#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
from zoo import fuck_slim
from tf_utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('feature_channels', 64, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
flags.DEFINE_boolean('patch_slim', False, '')

flags.DEFINE_integer('stride', 4, '')
flags.DEFINE_integer('radius', 25, '')

flags.DEFINE_float('offset_weight', 1, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF


def params_loss (dxy, dxy_gt):
    l1 = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    return tf.reduce_sum(l1, axis=1)

class Model (aardvark.Model):

    def __init__ (self):
        super().__init__()
        if FLAGS.classes > 1:
            aardvark.print_red("Classes should be number of point classes,")
            aardvark.print_red("not counting background.  Usually 1.")
        pass

    def extra_stream_config (self, is_training):

        augments = aardvark.load_augments(is_training)
        shift = 0
        if is_training:
            shift = FLAGS.clip_shift

        return {
              "annotate": [1],
              "transforms": [{"type": "resize", "max_size": FLAGS.max_size}
                  ] + augments + [
                  {"type": "clip", "shift": shift, "width": FLAGS.fix_width, "height": FLAGS.fix_height, "round": FLAGS.clip_stride},
                  {"type": "keypoints.basic", 'downsize': FLAGS.stride, 'classes': FLAGS.classes, 'radius': FLAGS.radius},
                  #{"type": "anchors.dense.point", 'downsize': FLAGS.stride, 'lower_th': anchor_th, 'upper_th': anchor_th},
                  {"type": "drop"}, # remove original annotation 
                  ]
             }

    def feed_dict (self, record, is_training = True):
        _, images, _, mask, offsets = record
        return {self.is_training: is_training,
                self.images: images,
                self.mask: mask,
                self.gt_offsets: offsets}

    def build_graph (self):
        if True:    # setup inputs
            # parameters
            is_training = tf.placeholder(tf.bool, name="is_training")
            images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
            # the reset are for training only
            mask = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.classes))
            gt_offsets = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.classes*2))

            self.is_training = is_training
            self.images = images
            self.mask = mask
            self.gt_offsets = gt_offsets

        if True:    # setup backbone
            global PIXEL_MEANS
            fuck_slim.extend()
            if FLAGS.patch_slim:
                fuck_slim.patch(is_training)
            network_fn = nets_factory.get_network_fn(FLAGS.backbone, num_classes=None,
                        weight_decay=FLAGS.weight_decay, is_training=is_training)
            backbone, _ = network_fn(images-PIXEL_MEANS[:,:,:,:FLAGS.channels], global_pool=False, output_stride=FLAGS.backbone_stride, scope=FLAGS.backbone)

        with tf.variable_scope('head'), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':is_training}, padding='SAME'):

            if FLAGS.finetune:
                backbone = tf.stop_gradient(backbone)
            #net = slim_multistep_upscale(net, FLAGS.backbone_stride / FLAGS.stride, FLAGS.reduction)
            #backbone = net
            stride = FLAGS.backbone_stride // FLAGS.stride
            #backbone = slim.conv2d_transpose(backbone, FLAGS.feature_channels, st*2, st)

            #prob = slim.conv2d(backbone, FLAGS.classes, 3, 1, activation_fn=tf.sigmoid) 
            prob = slim.conv2d_transpose(backbone, FLAGS.classes, stride*2, stride, activation_fn=tf.sigmoid)

            #logits2 = tf.reshape(logits, (-1, 2))
            #prob2 = tf.squeeze(tf.slice(tf.nn.softmax(logits2), [0, 1], [-1, 1]), 1)
            #tf.reshape(prob2, tf.shape(mask), name='prob')
            #xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=mask)
            dice = tf.identity(dice_loss(mask, prob), name='di')

            tf.losses.add_loss(dice)
            self.metrics.append(dice)

            offsets = slim.conv2d_transpose(backbone, FLAGS.classes*2, stride*2, stride, activation_fn=None)
            offsets2 = tf.reshape(offsets, (-1, 2))     # ? * 4
            gt_offsets2 = tf.reshape(gt_offsets, (-1,2))
            mask2 = tf.reshape(mask, (-1,))

            pl = params_loss(offsets2, gt_offsets2) * mask2
            pl = tf.reduce_sum(pl) / (tf.reduce_sum(mask2) + 1)
            pl = tf.check_numerics(pl * FLAGS.offset_weight, 'pl', name='p1') # params-loss

            tf.losses.add_loss(pl)
            self.metrics.append(pl)
        tf.identity(prob, name='prob')
        tf.identity(offsets, 'offsets')

        if FLAGS.finetune:
            assert FLAGS.colorspace == 'RGB'
            def is_trainable (x):
                return x.startswith('head')
            self.init_session, self.variables_to_train = aardvark.setup_finetune(FLAGS.finetune, is_trainable)
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

