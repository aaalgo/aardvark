#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory
import aardvark

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('net', 'resnet_v2_50', 'architecture')


class Model (aardvark.ClassificationModel):
    def __init__ (self):
        super().__init__()
        pass

    def inference (self, images, classes, is_training):

        logits = aardvark.create_stock_slim_network(FLAGS.net, images, is_training, num_classes=classes, global_pool=True)

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

