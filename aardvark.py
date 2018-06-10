#!/usr/bin/env python4

# This is the basic aaalgo tensorflow model training framework.

import errno
import os
from abc import ABC, abstractmethod
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# C++ code, python3 setup.py build
import time, datetime
import logging
import simplejson as json
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import picpac

flags = tf.app.flags
FLAGS = flags.FLAGS

# PicPac-related parameters
flags.DEFINE_string('db', None, 'training db')
flags.DEFINE_string('val_db', None, 'validation db')

flags.DEFINE_string('mixin', None, 'db to be mixed into training')
flags.DEFINE_integer('channels', 3, 'image channels')
flags.DEFINE_boolean('cache', True, 'cache images in memory') # cache small db in memory
flags.DEFINE_string('augments', 'augments.json', 'augment config file')
flags.DEFINE_string('colorspace', 'RGB', 'colorspace')
flags.DEFINE_integer('picpac_dump', 20, 'dump training example for debugging')

flags.DEFINE_integer('max_size', 2000, 'max image size') 
flags.DEFINE_integer('batch', 1, 'batch size')

# model saving parameters
flags.DEFINE_string('model', 'model', 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_integer('max_to_keep', 100, 'models to keep')
flags.DEFINE_integer('epoch_steps', None, 'by default all images')
flags.DEFINE_integer('max_epochs', 500, '')
flags.DEFINE_integer('ckpt_epochs', 10, '')
flags.DEFINE_integer('val_epochs', 10, '')

# optimizer settings
flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 500, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
flags.DEFINE_boolean('adam', True, '')


def load_augments (is_training):
    augments = []
    if is_training:
        if FLAGS.augments:
            with open(FLAGS.augments, 'r') as f:
                augments = json.loads(f.read())
            print("Using augments:")
            print(json.dumps(augments))
            pass
        pass
    return augments

def create_picpac_stream (path, is_training, extra_config):

    assert os.path.exists(path)
    print("CACHE:", FLAGS.cache)
    # check db size, warn not to cache if file is big
    statinfo = os.stat(path)
    if statinfo.st_size > 0x40000000 and FLAGS.cache:
        print_red("DB is probably too big too be cached, consider adding --cache 0")

    config = {"db": path,
              "loop": is_training,
              "shuffle": is_training,
              "reshuffle": is_training,
              "annotate": [],
              "channels": FLAGS.channels,
              "stratify": is_training,
              "dtype": "float32",
              "batch": FLAGS.batch,
              "colorspace": FLAGS.colorspace,
              "cache": FLAGS.cache,
              "transforms": []
             }

    if is_training:
        config["dump"] = FLAGS.picpac_dump # dump 20 training samples for debugging and see

    if is_training and not FLAGS.mixin is None:
        print("mixin support is incomplete in new picpac.")
        assert os.path.exists(FLAGS.mixin)
        config['mixin'] = FLAGS.mixin
        config['mixin_group_reset'] = 0
        config['mixin_group_delta'] = 1
        pass
    config.update(extra_config)
    return picpac.ImageStream(config)

class Model(ABC):
    def __init__ (self):
        super().__init__()
        self.metrics = []

    @abstractmethod
    def build_graph (self):
        pass

    def extra_stream_config (self, is_training):
        return {}

    @abstractmethod
    def feed_dict (self, record):
        pass
    pass

class Metrics:  # display metrics
    def __init__ (self, model):
        self.metric_names = [x.name[:-2] for x in model.metrics]
        self.cnt, self.sum = 0, np.array([0] * len(model.metrics), dtype=np.float32)
        pass

    def update (self, mm, cc):
        self.sum += np.array(mm) * cc
        self.cnt += cc
        self.avg = self.sum / self.cnt
        return ' '.join(['%s=%.3f' % (a, b) for a, b in zip(self.metric_names, list(self.avg))])

def train (model):

    logging.basicConfig(filename='train-%s-%s.log' % (type(model).__name__, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')),level=logging.DEBUG, format='%(asctime)s %(message)s')

    if FLAGS.model:
        try:    # create directory if not exists
            os.makedirs(FLAGS.model)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    model.build_graph()

    global_step = tf.train.create_global_step()
    LR = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    if FLAGS.adam:
        print("Using Adam optimizer, reducing LR by 100x")
        optimizer = tf.train.AdamOptimizer(LR/100)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.9)

    metrics = model.metrics
    reg_loss = tf.add_n(tf.losses.get_regularization_losses(), name='l2')
    for loss in tf.losses.get_losses():
        print("LOSS:", loss.name)
    total_loss = tf.losses.get_total_loss(name='L')
    metrics.extend([reg_loss, total_loss])

    train_op = tf.contrib.training.create_train_op(total_loss, optimizer, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    stream = create_picpac_stream(FLAGS.db, True, model.extra_stream_config(True))
    # load validation db
    val_stream = None
    if FLAGS.val_db:
        val_stream = create_picpac_stream(FLAGS.val_db, False, model.extra_stream_config(False))

    epoch_steps = FLAGS.epoch_steps
    if epoch_steps is None:
        epoch_steps = (stream.size() + FLAGS.batch-1) // FLAGS.batch
    best = 0

    ss_config = tf.ConfigProto()
    ss_config.gpu_options.allow_growth=True

    with tf.Session(config=ss_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)

        global_start_time = time.time()
        epoch, step = 0, 0

        while epoch < FLAGS.max_epochs:
            start_time = time.time()
            metrics = Metrics(model)
            progress = tqdm(range(epoch_steps), leave=False)
            for _ in progress:
                record = stream.next()
                mm, _ = sess.run([model.metrics, train_op], feed_dict=model.feed_dict(record, True))
                metrics_txt = metrics.update(mm, record[1].shape[0])
                progress.set_description(metrics_txt)
                step += 1
                pass
            stop = time.time()
            msg = 'train epoch=%d step=%d %s elapsed=%.3f time=%.3f' % (
                        epoch, step, metrics_txt, stop - global_start_time, stop - start_time)
            print_green(msg)
            logging.info(msg)

            epoch += 1

            if (epoch % FLAGS.val_epochs == 0) and val_stream:
                lr = sess.run(LR)
                # evaluation
                metrics = Metrics(model)
                val_stream.reset()
                progress = tqdm(val_stream, leave=False)
                for record in progress:
                    mm = sess.run(model.metrics, feed_dict=model.feed_dict(record, False))
                    metrics_txt = metrics.update(mm, record[1].shape[0])
                    progress.set_description(metrics_txt)
                    pass
                if metrics.avg[-1] > best:
                    best = metrics.avg[-1]
                msg = 'valid epoch=%d step=%d %s lr=%.4f best=%.3f' % (
                            epoch-1, step, metrics_txt, lr, best)
                print_red(msg)
                logging.info(msg)
            # model saving
            if (epoch % FLAGS.ckpt_epochs == 0) and FLAGS.model:
                ckpt_path = '%s/%d' % (FLAGS.model, epoch)
                saver.save(sess, ckpt_path)
                print('saved to %s.' % ckpt_path)
            pass
        pass
    pass

def print_red (txt):
    print('\033[91m' + txt + '\033[0m')

def print_green (txt):
    print('\033[92m' + txt + '\033[0m')

