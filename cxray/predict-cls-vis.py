#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('..')
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from mold import Scaling
from gallery import Gallery
from chest import *

class Model:
    def __init__ (self, X, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        is_training = tf.constant(False)
        self.probs, self.heatmap = \
                tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['probs:0', 'heatmap:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_integer('channels', 1, '')
flags.DEFINE_string('list', 'scratch/val-nz.list', '')
flags.DEFINE_integer('max', 10, '')
flags.DEFINE_integer('resize', 256, '')

def save_prediction_image (gal, image, label, probs, heatmap):
    pred = np.argmax(probs)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32) 
    orig = np.copy(image)

    # ground truth
    cv2.putText(image, 'gt %d: %.3f %s' % (label, probs[label], CATEGORIES[label][1]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, 'inf %d: %.3f %s' % (pred, probs[pred], CATEGORIES[pred][1]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    image[:, :, 1] += heatmap[:, :, label] * 128
    image[:, :, 2] += heatmap[:, :, pred] * 128
    image = np.concatenate([image, orig], axis=1)
    cv2.imwrite(gal.next(), np.clip(image, 0, 255))
    pass

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    mold = Scaling(stride = FLAGS.stride)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)

        gal = Gallery('output', ext='.png')
        CC = 0
        if FLAGS.list:
            with open(FLAGS.list, 'r') as f:
                for line in f:
                    if CC > FLAGS.max:
                        break
                    path, label = line.strip().split(',')
                    label = int(label)

                    print(path)
                    if FLAGS.channels == 3:
                        image = cv2.imread(path, cv2.IMREAD_COLOR)
                    elif FLAGS.channels == 1:
                        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    else:
                        assert False

                    image = cv2.resize(image, (FLAGS.resize, FLAGS.resize))

                    probs, heatmap = sess.run([model.probs, model.heatmap], feed_dict={X: mold.batch_image(image)})
                    probs = probs[0]
                    heatmap = mold.unbatch_prob(image, heatmap)
                    '''END INFERENCE'''

                    save_prediction_image(gal, image, label, probs, heatmap)
                    CC += 1
        gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()

