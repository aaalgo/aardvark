#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
from tf_utils import *
from zoo import fuck_slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
flags.DEFINE_boolean('patch_slim', False, '')
flags.DEFINE_integer('reduction', 1, '')
flags.DEFINE_integer('multistep', 0, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF SLIM

class Model (aardvark.SegmentationModel):
    def __init__ (self):
        super().__init__()
        pass

    def inference (self, images, classes, is_training):
        assert FLAGS.clip_stride % FLAGS.backbone_stride == 0
        global PIXEL_MEANS

        fuck_slim.extend()
        if FLAGS.patch_slim:
            fuck_slim.patch(is_training)

        network_fn = nets_factory.get_network_fn(FLAGS.backbone, num_classes=None,
                        weight_decay=FLAGS.weight_decay, is_training=is_training)

        backbone, _ = network_fn(images-PIXEL_MEANS, global_pool=False, output_stride=FLAGS.backbone_stride, scope=FLAGS.backbone)

        if FLAGS.finetune:
            backbone = tf.stop_gradient(backbone)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':is_training}):
            if FLAGS.multistep > 0:
                if FLAGS.multistep == 1:
                    aardvark.print_red("multistep = 1 doesn't converge well")
                net = slim_multistep_upscale(backbone, FLAGS.backbone_stride, FLAGS.reduction, FLAGS.multistep)
                logits = slim.conv2d(net, classes, 3, 1, activation_fn=None, padding='SAME')
            else:
                logits = slim.conv2d_transpose(backbone, classes, FLAGS.backbone_stride * 2, FLAGS.backbone_stride, activation_fn=None, padding='SAME')
        if FLAGS.finetune:
            assert FLAGS.colorspace == 'RGB'
            def is_trainable (x):
                return not x.startswith(FLAGS.backbone)
            self.init_session, self.variables_to_train = aardvark.setup_finetune(FLAGS.finetune, is_trainable)
        return logits

def main (_):
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

