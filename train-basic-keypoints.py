#!/usr/bin/env python3
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/research/slim'))
# C++ code, python3 setup.py build
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('classes', 1, 'number of classes')

flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('stride', 4, '')
flags.DEFINE_integer('radius', 25, '')
flags.DEFINE_integer('features', 128, '')
flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')

flags.DEFINE_float('pos_weight', 1, '')
flags.DEFINE_float('offset_weight', 1, '')
flags.DEFINE_float('re_weight', 1, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF

def create_picpac_stream (db_path, is_training):
    #anchor_th = math.exp(-FLAGS.radius)

def tf_repeat(tensor, repeats):
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

def params_loss (dxy, dxy_gt):
    l1 = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE)
    return tf.reduce_sum(l1, axis=1)

class KpModel (aardvark.Model):
    def __init__ (self):
        super().__init__()
        pass

    def feed_dict (self, record, is_training = True):
        _, images, _, mask, offsets = record
        return {self.is_training: is_training,
                self.images: images,
                self.mask: mask,
                self.gt_offsets: offsets}

    def build_graph (self):
        if True:    # setup inputs
            # parameters
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
            # the reset are for training only
            self.mask = tf.placeholder(tf.int32, shape=(None, None, None, FLAGS.classes))
            self.gt_offsets = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.classes*2))

        if True:    # setup backbone
            global PIXEL_MEANS
            if not FLAGS.finetune:
                patch_arg_scopes()
            network_fn = nets_factory.get_network_fn(FLAGS.backbone, num_classes=None,
                        weight_decay=FLAGS.weight_decay, is_training=self.is_training)

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'), \
                 slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':self.is_training}), \
                 slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                net, _ = network_fn(self.images-PIXEL_MEANS, global_pool=False, output_stride=FLAGS.backbone_stride)
                assert FLAGS.backbone_stride % FLAGS.stride == 0
                ss = FLAGS.backbone_stride // FLAGS.stride
                self.backbone = slim.conv2d_transpose(net, FLAGS.features, ss*2, ss)
                pass

        if True:    # setup regression
            logits = slim.conv2d(self.backbone, 2 * FLAGS.classes, 3, 1, activation_fn=None) 
            logits = tf.check_numerics(logits, 'logits')
            logits2 = tf.reshape(logits, (-1, 2))
            prob2 = tf.squeeze(tf.slice(tf.nn.softmax(logits2), [0, 1], [-1, 1]), 1)
            tf.reshape(prob2, tf.shape(self.mask), name='prob')
            mask = tf.reshape(self.mask, (-1, ))
            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=mask)

            mask = tf.cast(mask, tf.float32)
            w = (FLAGS.pos_weight-1.0) * mask + 1.0
            xe = tf.reduce_sum(xe * w) / (tf.reduce_sum(w) + 1)
            xe = tf.check_numerics(xe, 'xe', name='xe')    # rpn xe

            tf.losses.add_loss(xe)
            self.metrics.append(xe)

            offsets = slim.conv2d(self.backbone, 2 * FLAGS.classes, 3, 1, activation_fn=None)
            self.offsets = tf.identity(offsets, 'offsets')
            offsets2 = tf.reshape(offsets, (-1, 2))     # ? * 4
            gt_offsets2 = tf.reshape(self.gt_offsets, (-1,2))

            pl = params_loss(offsets2, gt_offsets2) * mask
            pl = tf.reduce_sum(pl) / (tf.reduce_sum(mask) + 1)
            pl = tf.check_numerics(pl * FLAGS.offset_weight, 'pl', name='p1') # params-loss

            tf.losses.add_loss(pl)
            self.metrics.append(pl)
        pass

    def stream_config (self, is_training):
        augments = aardvark.load_augments(is_training)

        return {
              "annotate": [1],
              "transforms": augments + [
                  {"type": "resize", "max_size": FLAGS.max_size},
                  {"type": "clip", "round": FLAGS.backbone_stride},
                  {"type": "keypoints.basic", 'downsize': FLAGS.stride, 'classes': FLAGS.classes, 'radius': FLAGS.radius},
                  #{"type": "anchors.dense.point", 'downsize': FLAGS.stride, 'lower_th': anchor_th, 'upper_th': anchor_th},
                  {"type": "drop"}, # remove original annotation 
                  ]
             }

def main (_):
    model = KpModel()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

