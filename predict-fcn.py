#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
from gallery import Gallery

class Model:
    def __init__ (self, X, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        is_training = tf.constant(False)
        self.prob, = \
                tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['prob:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('clip_stride', 16, '')
flags.DEFINE_integer('max_size', 2000, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_string('colorspace', 'RGB', '')

flags.DEFINE_string('db', None, '')
flags.DEFINE_string('list', None, '')
flags.DEFINE_integer('max', 50, '')


def save_prediction_image (gal, image, prob):
    cv2.imwrite(gal.next(), image)

    label = np.copy(image).astype(np.float32)
    label *= 0
    label[:, :, 0] += prob[:, :] * 255
    cv2.imwrite(gal.next(), np.clip(label, 0, 255))
    pass

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)

        gal = Gallery('output', cols=2, ext='.jpg')
        CC = 0
        if FLAGS.list:
            with open(FLAGS.list, 'r') as f:
                for path in f:
                    if CC > FLAGS.max:
                        break
                    path = path.strip()
                    print(path)
                    if FLAGS.channels == 3:
                        image = cv2.imread(path, cv2.IMREAD_COLOR)
                    elif FLAGS.channels == 1:
                        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        image = np.expand_dims(image, axis=3)
                    else:
                        assert False
                    H, W = image.shape[:2]

                    if max(H, W) > FLAGS.max_size:
                        f = FLAGS.max_size / max(H, W)
                        image = cv2.resize(image, None, fx=f, fy=f)
                        H, W = image.shape[:2]
                    '''BEGIN INFERENCE'''
                    # clip edge
                    H = H // FLAGS.clip_stride * FLAGS.clip_stride
                    W = W // FLAGS.clip_stride * FLAGS.clip_stride
                    image = image[:H, :W].astype(np.float32)
                    # change from BGR to RGB
                    if FLAGS.channels == 3 and FLAGS.colorspace == 'RGB':
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        batch = np.expand_dims(image_rgb, axis=0)
                    else:
                        batch = np.expand_dims(image, axis=0)
                    prob = sess.run(model.prob, feed_dict={X: batch})
                    '''END INFERENCE'''
                    save_prediction_image(gal, image, prob[0])
                    CC += 1
        if FLAGS.db:
            stream = picpac.ImageStream({'db': FLAGS.db, 'loop': False, 'channels': FLAGS.channels, 'colorspace': FLAGS.colorspace, 'threads': 1, 'shuffle': False,
                                         'transforms': [{"type": "resize", "max_size": FLAGS.max_size},
                                                        {"type": "clip", "round": FLAGS.clip_stride}]})
            for meta, batch in stream:
                    if CC > FLAGS.max:
                        break
                    print(meta.ids)
                    image = batch[0]
                    if FLAGS.channels == 3 and FLAGS.colorspace == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    prob = sess.run(model.prob, feed_dict={X: batch})
                    '''END INFERENCE'''
                    save_prediction_image(gal, image, prob[0])
                    CC += 1
        gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()

