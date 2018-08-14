#!/usr/bin/env python3
import tensorflow as tf
import aardvark

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('re_weight', 0.0001, 'regularization weight')

class Model (aardvark.SegmentationModel):
    def __init__ (self):
        super().__init__()
        pass

    def inference (self, images, classes, is_training):
        self.backbone, backbone_stride = myunet(self.images-127.0, self.is_training)
        assert FLAGS.clip_stride % backbone_stride == 0
        return tf.layers.conv2d_transpose(self.backbone, classes, 3, 1, activation=None, padding='SAME')
    pass

def myunet (X, is_training):
    BN = True
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
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

