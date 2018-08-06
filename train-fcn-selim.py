#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/dsb_selim'))
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import aardvark
from models.model_factory import make_model

flags = tf.app.flags
flags.DEFINE_string('net', 'resnet50_2', 'architecture')
FLAGS = flags.FLAGS

def acc (a, b): # just for shorter name
    return keras.metrics.sparse_categorical_accuracy(a, b)


def prep (record):
    meta, images, labels = record
    return images, labels

def build_model ():
    assert FLAGS.fix_width > 0
    assert FLAGS.fix_height > 0
    model = make_model(FLAGS.net, [FLAGS.fix_height, FLAGS.fix_width, FLAGS.channels])
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=[acc])
    return model

def main (_):
    from keras.backend import set_image_data_format
    from keras.backend.tensorflow_backend import set_session
    set_image_data_format('channels_last')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    model = build_model()

    sm = aardvark.SegmentationModel()

    train_stream = sm.create_stream(FLAGS.db, True)
    val_stream = sm.create_stream(FLAGS.val_db, False)
    # we neet to reset val_stream
    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: val_stream.reset()),
                 keras.callbacks.ModelCheckpoint(FLAGS.model, period=FLAGS.ckpt_epochs),
                ]

    hist = model.fit_generator(map(prep, train_stream),
                                    steps_per_epoch=train_stream.size()//FLAGS.batch,
                                    epochs=FLAGS.max_epochs,
                                    validation_data=map(prep, val_stream),
                                    validation_steps=val_stream.size()//FLAGS.batch,
                                    callbacks=callbacks)
    model.save_weights(FLAGS.model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

