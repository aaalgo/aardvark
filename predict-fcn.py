#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from gallery import gallery

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
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_integer('max_size', 2000, '')

def save_prediction_image (gal, image, prob):
    cv2.imwrite(gal.next(), image)

    label = np.copy(image).astype(np.float32)
    label *= 0
    label[:, :, 0] += prob[:, :] * 255
    cv2.imwrite(gal.next(), np.clip(label, 0, 255))
    pass

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)

        gal = gallery.Gallery('output', cols=2, ext='.jpg')
        with open('val.txt', 'r') as f:
            for path in f:
                path = path.strip()
                print(path)
                image = cv2.imread(path, -1)
                H, W = image.shape[:2]

                if max(H, W) > FLAGS.max_size:
                    f = FLAGS.max_size / max(H, W)
                    image = cv2.resize(image, None, fx=f, fy=f)
                    H, W = image.shape[:2]
                '''BEGIN INFERENCE'''
                # clip edge
                H = H // FLAGS.stride * FLAGS.stride
                W = W // FLAGS.stride * FLAGS.stride
                image = image[:H, :W, :].astype(np.float32)
                # change from BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                batch = np.expand_dims(image_rgb, axis=0)
                prob = sess.run(model.prob, feed_dict={X: batch})
                '''END INFERENCE'''
                save_prediction_image(gal, image, prob[0])
            gal.flush()
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

