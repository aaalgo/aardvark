#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory
import aardvark
from zoo import fuck_slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('net', 'resnet_v2_50', 'architecture')
flags.DEFINE_float('weight_decay', 0.00004, '')
flags.DEFINE_boolean('patch_slim', False, '')


class Model (aardvark.ClassificationModel):
    def __init__ (self):
        super().__init__()
        pass

    def inference (self, images, classes, is_training):
        PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF SLIM

        fuck_slim.extend()
        if FLAGS.patch_slim:
            fuck_slim.patch(is_training)
        
        network_fn = nets_factory.get_network_fn(FLAGS.net, num_classes=classes, is_training=is_training, weight_decay=FLAGS.weight_decay)

        logits, _ = network_fn(images-PIXEL_MEANS, scope=FLAGS.net)

        if FLAGS.finetune:
            assert FLAGS.colorspace == 'RGB'
            self.init_session, self.variables_to_train = aardvark.setup_finetune(FLAGS.finetune, lambda x: 'logits' in x)
        return logits
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

