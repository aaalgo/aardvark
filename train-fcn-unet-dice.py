#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import aardvark

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_float('re_weight', 0.0001, 'regularization weight')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF SLIM

def dice_loss (gt, prob):
    return -2 * (tf.reduce_sum(gt * prob) + 0.00001) / (tf.reduce_sum(gt) + tf.reduce_sum(prob) + 0.00001)

class FcnModel (aardvark.Model):
    def __init__ (self):
        super().__init__()
        pass

    def extra_stream_config (self, is_training):
        augments = aardvark.load_augments(is_training)
        
        return {"annotate": [1],
                "transforms": [
                  {"type": "resize", "max_size": FLAGS.max_size},
                  ] + augments + [
                  # clip edge so width & height are divisible by stride (16)
                  #{"type": "clip", "round": FLAGS.backbone_stride},
                  {"type": "clip", "min_width": 1280, "max_width": 1280, "min_height": 800, "max_height": 800, "round": self.backbone_stride},
                  ]
             }

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        self.gt_labels = tf.placeholder(tf.int32, shape=(None, None, None, 1))
        global PIXEL_MEANS
        la = tf.squeeze(self.gt_labels, axis=3)  # [?,?,?,1] -> [?,?,?], picpac generates 4-D tensor

        la_float = tf.cast(la, dtype=tf.float32)

        self.backbone, self.backbone_stride = myunet(self.images-PIXEL_MEANS, self.is_training)

        prob = tf.layers.conv2d_transpose(self.backbone, 1, 3, 1, activation=tf.sigmoid, padding='SAME')
        self.prob = tf.identity(tf.squeeze(prob, axis=3), name='prob')
        #self.prob = tf.squeeze(tf.slice(tf.nn.softmax(self.logits), [0, 0, 0, 1], [-1, -1, -1, -1]), axis=3, name='prob')

        dice = tf.identity(dice_loss(la_float, self.prob), name='di')

        tf.losses.add_loss(dice)
        self.metrics.append(dice)
        pass

    def feed_dict (self, record, is_training = True):
        # load picpac record into feed_dict
        _, images, labels = record
        return {self.is_training: is_training,
                self.images: images,
                self.gt_labels: labels}

def myunet (X, is_training):
    BN = False
    net = X
    stack = []
    with tf.name_scope('myunet'):
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.re_weight)

        def conv2d (input, channels, filter_size, stride):
            if BN:
                input = tf.layers.conv2d(input, channels, filter_size, stride, padding='SAME', activation=None, kernel_regularizer=regularizer)
                input = tf.layers.batch_normalization(input, training=is_training)
                return tf.nn.relu(input)
            return tf.layers.conv2d(input, channels, filter_size, stride, padding='SAME', kernel_regularizer=regularizer)

        def max_pool2d (input, filter_size, stride):
            return tf.layers.max_pooling2d(input, filter_size, stride, padding='SAME')

        def conv2d_transpose (input, channels, filter_size, stride):
            if BN:
                input = tf.layers.conv2d_transpose(input, channels, filter_size, stride, padding='SAME', activation=None, kernel_regularizer=regularizer)
                input = tf.layers.batch_normalization(input, training=is_training)
                return tf.nn.relu(input)
            return tf.layers.conv2d_transpose(input, channels, filter_size, stride, padding='SAME', kernel_regularizer=regularizer)

        net = conv2d(net, 32, 3, 2)
        net = conv2d(net, 32, 3, 1)
        stack.append(net)       # 1/2
        net = conv2d(net, 64, 3, 1)
        net = conv2d(net, 64, 3, 1)
        net = max_pool2d(net, 2, 2)
        stack.append(net)       # 1/4
        net = conv2d(net, 128, 3, 1)
        net = conv2d(net, 128, 3, 1)
        net = max_pool2d(net, 2, 2)
        stack.append(net)       # 1/8
        net = conv2d(net, 256, 3, 1)
        net = conv2d(net, 256, 3, 1)
        net = max_pool2d(net, 2, 2)
                                # 1/16
        net = conv2d(net, 256, 3, 1)
        net = conv2d(net, 256, 3, 1)
        net = conv2d_transpose(net, 128, 5, 2)
                                # 1/8
        net = tf.concat([net, stack.pop()], 3)
        net = conv2d_transpose(net, 64, 5, 2)
                                # 1/4
        net = tf.concat([net, stack.pop()], 3)
        net = conv2d_transpose(net, 32, 5, 2)
        net = tf.concat([net, stack.pop()], 3)
        net = conv2d_transpose(net, 16, 5, 2)
        assert len(stack) == 0
    return net, 16

def main (_):
    model = FcnModel()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

