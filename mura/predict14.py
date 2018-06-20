#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac

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
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_integer('channels', 1, '')
flags.DEFINE_integer('max_size', 400, '')
flags.DEFINE_integer('fix_size', 400, '')
flags.DEFINE_integer('max', 0, '')

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels))
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    stream = picpac.ImageStream({'db': 'scratch/val.db',
                                 'cache': False,
                                 'loop': False,
                                 'channels': FLAGS.channels,
                                 'shuffle': False,
                                 'batch': 1,
                                 'raw': [1],
                                 'colorspace': 'RGB',
                                 'transforms': [
                                    {"type": "resize", "max_size": FLAGS.max_size},
                                    {"type": "clip", "width": FLAGS.fix_size, "height": FLAGS.fix_size},
                                     ]})
                                                

    with tf.Session(config=config) as sess:
        model.loader(sess)
        lookup = {}
        C = 0
        for meta, batch in tqdm(stream, total=stream.size()):
            path = meta.raw[0][0].decode('ascii')
            probs = sess.run(model.probs, feed_dict={X: batch})
            case = '/'.join(path.split('/')[:-1]) + '/'
            if not case in lookup:
                lookup[case] = [0, np.zeros((14, ), dtype=np.float32)]
                pass
            case = lookup[case]
            case[0] = case[0] + 1
            case[1] += probs[0]
            C += 1
            if FLAGS.max > 0 and C >= FLAGS.max:
                break
            pass
        with open('predict.csv', 'w') as f, \
             open('predict.csv.full', 'w') as f2:

            for k, v in lookup.items():
                probs = np.reshape(v[1] / v[0], (7, 2))
                prob = np.sum(probs[:, 1])
                if prob > 0.5:
                    l = 1
                else:
                    l = 0
                f.write('%s,%d\n' % (k, l))
                f.write('%s,%g\n' % (k, prob))
                pass
    pass

if __name__ == '__main__':
    tf.app.run()

