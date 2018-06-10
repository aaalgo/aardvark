#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import picpac
import aardvark
from zoo import cls_nets as nets

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_integer('size', 224, '') 
flags.DEFINE_integer('shift', 0, '') 

flags.DEFINE_string('net', 'resnet_50', 'architecture')

class ClsModel(aardvark.Model):
    def __init__ (self):
        super().__init__()
        pass

    def build_graph (self):
        self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        self.labels = tf.placeholder(tf.int32, shape=(None, ))
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # load network
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(2.5e-4)), \
             slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=5e-4): 
            logits = getattr(nets, FLAGS.net)(self.images-127, self.is_training, FLAGS.classes)
            # probability of class 1 -- not very useful if FLAGS.classes > 2
            probs = tf.squeeze(tf.slice(tf.nn.softmax(logits), [0,1], [-1,1]), 1, name='prob')

        # cross-entropy
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        xe = tf.reduce_mean(xe, name='xe')
        # accuracy
        acc = tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32)
        acc = tf.reduce_mean(acc, name='acc')
        # regularization
        reg = tf.reduce_sum(tf.losses.get_regularization_losses(), name='re')
        # loss
        tf.losses.add_loss(xe)
        self.metrics.extend([xe, acc, reg, tf.losses.get_total_loss(name='lo')])
        pass

    def extra_stream_config (self, is_training):
        return { "transforms": aardvark.load_augments(is_training) + [
                      #{"type": "resize", "size": FLAGS.size},
                      {"type": "clip", "size": FLAGS.size, "shift": FLAGS.shift, "border_type": "replicate"},
                 ]}

    def feed_dict (self, record, is_training):
        meta, images = record
        return {self.images: images,
                self.labels: meta.labels,
                self.is_training: is_training}
    pass


def main (_):
    model = ClsModel()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

