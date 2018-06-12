#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
import time
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac, cpp

class Model:
    def __init__ (self, X, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.prob, self.offsets = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X,
                               'is_training:0': is_training},
                    return_elements=['prob:0', 'offsets:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('input_db', None, '')
flags.DEFINE_integer('stride', 4, '')
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('max', 50, '')

flags.DEFINE_float('anchor_th', 0.5, '')

def save_prediction_image (path, image, kp, mask, prob):
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    H, W = image.shape[:2]
    blue = image[:, :, 0]
    blue += 255 * cv2.resize(prob, (W, H))
    red = image[:, :, 2]
    mask = mask > 0
    red[mask] *= 0.5
    red[mask] += 127
    for x, y, c, score in kp:
        #if score < 5:
        #    continue
        cv2.circle(image, (x, y), 3, (0,255,0), 2)
        cv2.putText(image, '%.4f'%score, (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(path, np.clip(image, 0, 255))
    pass

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    model = Model(X, is_training, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model.loader(sess)
        if FLAGS.input:
            assert False
            '''
            assert os.path.exists(FLAGS.input)
            image = cv2.imread(FLAGS.input, cv2.IMREAD_COLOR)
            batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
            boxes, probs = sess.run([model.boxes, model.probs], feed_dict={X: batch, is_training: False})
            save_prediction_image(FLAGS.input + '.prob.png', image, boxes, probs)
            '''
        if FLAGS.input_db:
            assert os.path.exists(FLAGS.input_db)
            from gallery import Gallery
            picpac_config = {"db": FLAGS.input_db,
                      "loop": False,
                      "shuffle": False,
                      "reshuffle": False,
                      "annotate": False,
                      "channels": FLAGS.channels,
                      "colorspace": "RGB",
                      "stratify": False,
                      "dtype": "float32",
                      "batch": 1,
                      "annotate": [1],
                      "transforms": [
                          {"type": "clip", "round": FLAGS.backbone_stride},
                          {"type": "keypoints.basic", 'downsize': 1, 'classes': 1, 'radius': 25},
                          {"type": "drop"}, # remove original annotation 
                          ]
                     }
            stream = picpac.ImageStream(picpac_config)
            gal = Gallery('out')
            C = 0
            for meta, images, _, label, _ in stream:
                shape = list(images.shape)
                shape[1] //= FLAGS.stride
                shape[2] //= FLAGS.stride
                shape[3] = 1
                prob, offsets = sess.run([model.prob, model.offsets], feed_dict={X: images, is_training: False})
                kp = cpp.predict_basic_keypoints(prob[0], offsets[0], FLAGS.stride, 0.1)
                print(images.shape, prob.shape, offsets.shape, kp)
                save_prediction_image(gal.next(), images[0], kp, label[0, :, :, 0], prob[0, :, :, 0])
                C += 1
                if FLAGS.max and C >= FLAGS.max:
                    break
                pass
            pass
            gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()

