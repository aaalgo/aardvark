#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
from gallery import Gallery
import cv2
from glob import glob 

class Model:
    def __init__ (self, X, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        is_training = tf.constant(False, dtype=tf.bool)
        self.probs, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['probs:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('image', None, '')
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('max_size', 400, '')
flags.DEFINE_integer('fix_width', 200, '')
flags.DEFINE_integer('fix_height', 112, '')

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels))
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        model.loader(sess)
        gal_0 = Gallery('gallery_0',ext='png')
        gal_1 = Gallery('gallery_1',ext='png')
        gal_2 = Gallery('gallery_2',ext='png')
        for img in glob(os.path.join(FLAGS.image,"*/*.jpg")):
            filename = img.split("/")[-1]
            image = cv2.imread(img, cv2.IMREAD_COLOR)
            batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
            probs = sess.run(model.probs, feed_dict={X: batch})
            cls = np.argmax(probs[0])
            if cls == 0:
                cv2.imwrite(gal_0.next(filename=filename),image)
            if cls == 1:
                cv2.imwrite(gal_1.next(filename=filename),image)
            if cls == 2:
                cv2.imwrite(gal_2.next(filename=filename),image)
            '''
            if cls == 1:
                cv2.imwrite('gallery_1/'+filename,image)
                gal_1.next(filename=filename)
            if cls == 2:
                cv2.imwrite('gallery_2/'+filename,image)
                gal_2.next(filename=filename)
            '''
        gal_0.flush()
        gal_1.flush()
        gal_2.flush()

if __name__ == '__main__':
    tf.app.run()

