#!/usr/bin/env python3
import os
import sys
# C++ code, python3 setup.py build
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
from tf_utils import *
import cpp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('finetune', None, '')
flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('feature_channels', 64, '')
flags.DEFINE_integer('anchor_stride', 4, '')

flags.DEFINE_integer('rpn_channels', 128, '')
flags.DEFINE_integer('pooling_size', 7, '')
flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_integer('nms_max', 128, '')
flags.DEFINE_float('nms_th', 0.5, '')
flags.DEFINE_float('match_th', 0.5, '')
flags.DEFINE_integer('max_masks', 128, '')

# optimizer settings
flags.DEFINE_float('di_weight1', 1.0, '')
flags.DEFINE_float('pl_weight1', 1.0/50, '')
flags.DEFINE_float('xe_weight2', 1.0/50, '')
flags.DEFINE_float('pl_weight2', 1.0/50, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF

def anchors2boxes (shape, anchor_params, priors):
    # anchor parameters are: dx, dy, w, h
    B = shape[0]
    H = shape[1]
    W = shape[2]
    offset = tf_repeat(tf.range(B), [H * W * priors])
    if True:    # generate array of box centers
        x0 = tf.cast(tf.range(W) * FLAGS.anchor_stride, tf.float32)
        y0 = tf.cast(tf.range(H) * FLAGS.anchor_stride, tf.float32)
        x0, y0 = tf.meshgrid(x0, y0)
        x0 = tf.reshape(x0, (-1,))
        y0 = tf.reshape(y0, (-1,))
        x0 = tf.tile(tf_repeat(x0, [priors]), [B])
        y0 = tf.tile(tf_repeat(y0, [priors]), [B])
    dx, dy, lw, lh = [tf.squeeze(x, axis=1) for x in tf.split(anchor_params, [1,1,1,1], 1)]

    W = tf.cast(W * FLAGS.anchor_stride, tf.float32)
    H = tf.cast(H * FLAGS.anchor_stride, tf.float32)

    max_X = W-1
    max_Y = H-1

    w = tf.clip_by_value(tf.exp(lw)-1, 0, W)
    h = tf.clip_by_value(tf.exp(lh)-1, 0, H)

    x1 = x0 + dx - w/2
    y1 = y0 + dy - h/2
    x2 = x1 + w
    y2 = y1 + h
    x1 = tf.clip_by_value(x1, 0, max_X) 
    y1 = tf.clip_by_value(y1, 0, max_Y)
    x2 = tf.clip_by_value(x2, 0, max_X)
    y2 = tf.clip_by_value(y2, 0, max_Y)

    boxes = tf.stack([x1, y1, x2, y2], axis=1)
    return boxes, offset

def transform_bbox (roi, gt_box):
    x1,y1,x2,y2 = [tf.squeeze(x, axis=1) for x in tf.split(roi, [1,1,1,1], 1)]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    X1,Y1,X2,Y2 = [tf.squeeze(x, axis=1) for x in tf.split(gt_box, [1,1,1,1], 1)]
    W = X2 - X1 + 1
    H = Y2 - Y1 + 1
    CX = X1 + 0.5 * W
    CY = Y1 + 0.5 * H

    dx = (CX - cx) / w
    dy = (CY - cy) / h
    dw = W / w
    dh = H / h

    return tf.stack([dx, dy, dw, dh], axis=1)

def refine_bbox (roi, params):
    x1,y1,x2,y2 = [tf.squeeze(x, axis=1) for x in tf.split(roi, [1,1,1,1], 1)]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    dx = params[:, 0]
    dy = params[:, 1]
    dw = tf.exp(params[:, 2]) - 1
    dh = tf.exp(params[:, 3]) - 1

    CX = dx * w + cx
    CY = dy * h + cy
    W = dw * w
    H = dh * h

    return tf.stack([CX - 0.5 * W, CY - 0.5 * H, CX + 0.5 * W, CY + 0.5 * H], axis=1)

def normalize_boxes (shape, boxes):
    max_X = tf.cast(shape[2]-1, tf.float32)
    max_Y = tf.cast(shape[1]-1, tf.float32)
    x1,y1,x2,y2 = [tf.squeeze(x, axis=1) for x in tf.split(boxes, [1,1,1,1], 1)]
    x1 = x1 / max_X
    y1 = y1 / max_Y
    x2 = x2 / max_X
    y2 = y2 / max_Y
    return tf.stack([y1, x1, y2, x2], axis=1)

def shift_boxes (boxes, offset):
    assert FLAGS.batch == 1
    return boxes

def params_loss (params, gt_params):
    dxy, wh = tf.split(params, [2,2], 1)
    dxy_gt, wh_gt = tf.split(gt_params, [2,2], 1)
    wh_gt = tf.log(wh_gt + 1)
    l1 = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    l2 = tf.losses.huber_loss(wh, wh_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    return tf.reduce_sum(l1+l2, axis=1)

class Model (aardvark.Model):
    def __init__ (self):
        super().__init__()
        priors = 1
        self.priors = priors    # number of priors
        self.gt_matcher = cpp.GTMatcher(FLAGS.match_th, FLAGS.max_masks)
        pass

    def feed_dict (self, record, is_training = True):
        _, images, _, gt_anchors, gt_anchors_weight, gt_params, gt_params_weight, gt_boxes = record
        assert np.all(gt_anchors < 2)
        gt_boxes = np.reshape(gt_boxes, [-1, 7])
        if len(gt_boxes.shape) > 1:
            assert np.all(gt_boxes[:, 1] < FLAGS.classes)
            assert np.all(gt_boxes[:, 1] > 0)
        return {self.is_training: is_training,
                self.images: images,
                self.gt_anchors: gt_anchors,
                self.gt_anchors_weight: gt_anchors_weight,
                self.gt_params: gt_params,
                self.gt_params_weight: gt_params_weight,
                self.gt_boxes: gt_boxes}

    def build_graph (self):
        # Set up model inputs
        # parameters
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        anchor_th = tf.constant(FLAGS.anchor_th, dtype=tf.float32, name="anchor_th")
        nms_max = tf.constant(FLAGS.nms_max, dtype=tf.int32, name="nms_max")
        nms_th = tf.constant(FLAGS.nms_th, dtype=tf.float32, name="nms_th")

        # input images
        self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        # the reset are for training only
        # whether a location should be activated
        self.gt_anchors = tf.placeholder(tf.float32, shape=(None, None, None, self.priors))
        self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.priors))
        # parameter of that location
        self.gt_params = tf.placeholder(tf.float32, shape=(None, None, None, self.priors * 4))
        self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.priors))
        self.gt_boxes = tf.placeholder(tf.float32, shape=(None, 7))

        backbone = aardvark.create_stock_slim_network(FLAGS.backbone, self.images, self.is_training, global_pool=False, stride=FLAGS.backbone_stride)

        upscale = FLAGS.backbone_stride // FLAGS.anchor_stride
        with slim.arg_scope(aardvark.default_argscope(self.is_training)):
            
            # anchor activation signals
            anchors = slim.conv2d_transpose(backbone, self.priors, 2*upscale, upscale, activation_fn=tf.sigmoid)
            dice = tf.identity(weighted_dice_loss(self.gt_anchors, anchors, self.gt_anchors_weight) * FLAGS.di_weight1, name='di')

            tf.losses.add_loss(dice)
            self.metrics.append(dice)

            anchor_layer_shape = tf.shape(anchors)

            params = slim.conv2d_transpose(backbone, 4 * self.priors, 2*upscale, upscale, activation_fn=None)
            params = tf.reshape(params, (-1, 4))     # ? * 4
            gt_params = tf.reshape(self.gt_params, (-1, 4))
            gt_params_weight = tf.reshape(self.gt_params_weight, (-1,))

            params = tf.check_numerics(params, 'params')
            gt_params = tf.check_numerics(gt_params, 'gt_params')
            gt_params_weight = tf.check_numerics(gt_params_weight, 'gt_params_weight')

            pl = params_loss(params, gt_params) * gt_params_weight
            pl = tf.reduce_sum(pl) / (tf.reduce_sum(gt_params_weight) + 1)
            pl = tf.check_numerics(pl * FLAGS.pl_weight1, 'p1', name='p1') # params-loss

            tf.losses.add_loss(pl)
            self.metrics.append(pl)

            prob = tf.reshape(anchors, (-1,))
            # index is index within mini batch
            boxes, index = anchors2boxes(anchor_layer_shape, params, self.priors)

            with tf.device('/cpu:0'):
                # fuck tensorflow, these lines fail on GPU

                # pre-filtering by threshold so we put less stress on non_max_suppression
                sel = tf.greater_equal(prob, anchor_th)
                # sel is a boolean mask

                # select only boxes with prob > th for nms
                prob = tf.boolean_mask(prob, sel)
                #params = tf.boolean_mask(params, sel)
                boxes = tf.boolean_mask(boxes, sel)
                index = tf.boolean_mask(index, sel)

            sel = tf.image.non_max_suppression(shift_boxes(boxes, index), prob, nms_max, iou_threshold=nms_th)
            # sel is a list of indices
            rpn_probs = tf.gather(prob, sel, name='rpn_probs')
            rpn_boxes = tf.gather(boxes, sel, name='rpn_boxes')
            rpn_index = tf.gather(index, sel, name='rpn_index')

            n_hits, rpn_hits, gt_hits = tf.py_func(self.gt_matcher.apply, [rpn_boxes, rpn_index, self.gt_boxes], [tf.float32, tf.int32, tf.int32])

            # % boxes found
            precision = n_hits / (tf.cast(tf.shape(boxes)[0], tf.float32) + 0.001);
            recall = n_hits / (tf.cast(tf.shape(self.gt_boxes)[0], tf.float32) + 0.001);
            self.metrics.append(tf.identity(precision, name='p'))
            self.metrics.append(tf.identity(recall, name='r'))

            # setup prediction
            # normalize boxes to [0-1]
            boxes = normalize_boxes(tf.shape(self.images), rpn_boxes)

            mask_size = FLAGS.pooling_size * 2
            net = tf.image.crop_and_resize(backbone, boxes, rpn_index, [mask_size, mask_size])
            net = slim.max_pool2d(net, [2,2], padding='SAME')
            #
            net = slim.conv2d(net, FLAGS.feature_channels, 3, 1)
            net = tf.reshape(net, [-1, FLAGS.pooling_size * FLAGS.pooling_size * FLAGS.feature_channels])

            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training)

            logits = slim.fully_connected(net, FLAGS.classes, activation_fn=None)
            probs = tf.nn.softmax(logits, name='probs')

            cls = tf.argmax(logits, axis=1, name='cls')

            params = slim.fully_connected(net, FLAGS.classes * 4, activation_fn=None)
            params = tf.reshape(params, [-1, FLAGS.classes, 4])

            if True:    # for inference stage
                onehot = tf.expand_dims(tf.one_hot(tf.cast(cls, tf.int32), depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
                params_onehot = tf.reduce_sum(params * onehot, axis=1)
                boxes = refine_bbox(rpn_boxes, params_onehot)
                tf.identity(boxes, name='boxes')

            rpn_boxes = tf.gather(rpn_boxes, rpn_hits)
            logits = tf.gather(logits, rpn_hits)
            params = tf.gather(params, rpn_hits)

            matched_gt_boxes = tf.gather(self.gt_boxes, gt_hits)
            matched_gt_labels = tf.cast(tf.squeeze(tf.slice(matched_gt_boxes, [0, 1], [-1, 1]), axis=1), tf.int32)
            matched_gt_boxes = transform_bbox(rpn_boxes, tf.slice(matched_gt_boxes, [0, 3], [-1, 4]))

            onehot = tf.expand_dims(tf.one_hot(matched_gt_labels, depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
            params = tf.reduce_sum(params * onehot, axis=1)


            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=matched_gt_labels)
            xe = tf.check_numerics(tf.reduce_sum(xe)/(n_hits + 1)*FLAGS.xe_weight2, 'x2', name='x2')
            tf.losses.add_loss(xe)
            self.metrics.append(xe)

            pl = params_loss(params, matched_gt_boxes) 
            pl = tf.reduce_sum(pl) / (n_hits + 1)
            pl = tf.check_numerics(pl * FLAGS.pl_weight2, 'p2', name='p2') # params-loss
            tf.losses.add_loss(pl)
            self.metrics.append(pl)
        pass

    def extra_stream_config (self, is_training):
        augments = aardvark.load_augments(is_training)
        return {
                  "annotate": [1],
                  "transforms": 
                      [{"type": "resize", "max_size": FLAGS.max_size}] +
                      augments + [
                      {"type": "clip", "round": FLAGS.backbone_stride},
                      {"type": "anchors.dense.box", 'downsize': FLAGS.anchor_stride},
                      {"type": "box_feature"},
                      {"type": "rasterize"},
                      ]
                 }

def main (_):
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

