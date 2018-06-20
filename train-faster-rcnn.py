#!/usr/bin/env python3
import os
import math
import sys
# C++ code, python3 setup.py build
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
import cv2
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

flags.DEFINE_float('lower_th', 0.1, '')
flags.DEFINE_float('upper_th', 0.5, '')

# optimizer settings
flags.DEFINE_float('di_weight1', 1.0, '')
flags.DEFINE_float('pl_weight1', 1.0, '')
flags.DEFINE_float('xe_weight2', 1.0, '')
flags.DEFINE_float('pl_weight2', 1.0, '')

flags.DEFINE_boolean('rpnonly', False, '')
flags.DEFINE_boolean('rcnnonly', False, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF

def params_loss_rpn (params, gt_params, priors_wh):
    rel_dxy, log_wh = tf.split(params, [2,2], 1)
    # rel_xy = [x, y] / priors_wh

    dxy_gt, wh_gt = tf.split(gt_params, [2,2], 1)
    log_wh_gt = tf.log(wh_gt)

    dxy_gt = tf.reshape(dxy_gt, (-1, priors_wh.shape[1], 2))

    rel_dxy_gt = tf.reshape(dxy_gt / priors_wh, tf.shape(rel_dxy))

    #d_rel_xy_gt = dxy_gt / tf.exp(tf.clip_by_value(log_wh, 0, math.log(FLAGS.max_size)))

    l1 = tf.losses.huber_loss(rel_dxy, rel_dxy_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    l2 = tf.losses.huber_loss(log_wh, log_wh_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    return tf.reduce_sum(l1+l2, axis=1)

def anchors2boxes (shape, anchor_params, priors_wh):
    # anchor parameters are: dx, dy, w, h
    B = shape[0]
    H = shape[1]
    W = shape[2]
    n_priors = priors_wh.shape[1]
    offset = tf_repeat(tf.range(B), [H * W * n_priors])
    if True:    # generate array of box centers
        x0 = tf.cast(tf.range(W) * FLAGS.anchor_stride, tf.float32)
        y0 = tf.cast(tf.range(H) * FLAGS.anchor_stride, tf.float32)
        x0, y0 = tf.meshgrid(x0, y0)
        x0 = tf.reshape(x0, (-1,))
        y0 = tf.reshape(y0, (-1,))
        x0 = tf.tile(tf_repeat(x0, [n_priors]), [B])
        y0 = tf.tile(tf_repeat(y0, [n_priors]), [B])
    drxy, lw, lh = tf.split(anchor_params, [2,1,1], 1)
    drxy = tf.reshape(drxy, (-1, n_priors, 2))
    dxy = tf.reshape(drxy * priors_wh, (-1, 2))
    dx, dy = tf.split(dxy, [1,1], 1)
    dx = tf.squeeze(dx, axis=1)
    dy = tf.squeeze(dy, axis=1)
    lw = tf.squeeze(lw, axis=1)
    lh = tf.squeeze(lh, axis=1)

    W = tf.cast(W * FLAGS.anchor_stride, tf.float32)
    H = tf.cast(H * FLAGS.anchor_stride, tf.float32)

    max_X = W-1
    max_Y = H-1

    w = tf.clip_by_value(tf.exp(lw), 0, W)
    h = tf.clip_by_value(tf.exp(lh), 0, H)

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
    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    X1 = gt_box[:, 0]
    Y1 = gt_box[:, 1]
    X2 = gt_box[:, 2]
    Y2 = gt_box[:, 3]

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
    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    dx = params[:, 0]
    dy = params[:, 1]
    dw = tf.exp(params[:, 2])
    dh = tf.exp(params[:, 3])

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
    # boxes     N * [x1, y1, x2, y2]
    # offsets   N
    offset = tf.expand_dims(offset * FLAGS.max_size * 2, axis=1)
    # offset    N * [V]
    # such that there's no way for boxes from different offset to overlap
    return boxes + tf.cast(offset, dtype=tf.float32)


def params_loss (params, gt_params):
    dxy, log_wh = tf.split(params, [2,2], 1)
    dxy_gt, wh_gt = tf.split(gt_params, [2,2], 1)
    log_wh_gt = tf.check_numerics(tf.log(wh_gt), name='log_wh_gt', message='xxx')
    l1 = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    l2 = tf.losses.huber_loss(log_wh, log_wh_gt, reduction=tf.losses.Reduction.NONE, loss_collection=None)
    return tf.reduce_sum(l1+l2, axis=1)

dump_cnt = 0

class Model (aardvark.Model):
    def __init__ (self):
        super().__init__()
        self.gt_matcher = cpp.GTMatcher(FLAGS.match_th, FLAGS.max_masks, FLAGS.backbone_stride * 2)
        priors = []
        priors_wh = []
        if os.path.exists('priors'):
            with open('priors', 'r') as f:
                for l in f:
                    s, r = l.strip().split(' ')
                    s, r = float(s), float(r)
                    # w * h = s * s
                    # w / h = r
                    w = math.sqrt(s * s * r)
                    h = math.sqrt(s * s / r)
                    priors.append([s, r])
                    priors_wh.append([w, h])
                    pass
                pass
            pass
        self.priors = priors
        self.priors_wh = priors_wh
        if len(priors) > 0:
            self.n_priors = len(priors)
        else:
            self.n_priors = 1
        pass

    def feed_dict (self, record, is_training = True):
        global dump_cnt
        _, images, _, gt_anchors, gt_anchors_weight, gt_params, gt_params_weight, gt_boxes = record
        assert np.all(gt_anchors < 2)
        gt_boxes = np.reshape(gt_boxes, [-1, 7])
        if dump_cnt < 20:
            for i in range(images.shape[0]):
                cv2.imwrite('picpac_dump2/%d_image.png' % dump_cnt, images[i])
                for j in range(gt_anchors.shape[3]):
                    cv2.imwrite('picpac_dump2/%d_anchor_%d.png' % (dump_cnt, j), gt_anchors[i,:,:,j]*255)
                    cv2.imwrite('picpac_dump2/%d_mask_%d.png' % (dump_cnt, j), gt_anchors_weight[i,:,:,j]*255)
                    cv2.imwrite('picpac_dump2/%d_weight_%d.png' % (dump_cnt, j), gt_params_weight[i,:,:,j]*255)
                dump_cnt += 1


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
        self.gt_anchors = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        # parameter of that location
        self.gt_params = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors * 4))
        self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        self.gt_boxes = tf.placeholder(tf.float32, shape=(None, 7))
        self.priors_wh = tf.expand_dims(tf.constant(self.priors_wh, dtype=tf.float32), axis=0)

        backbone = aardvark.create_stock_slim_network(FLAGS.backbone, self.images, self.is_training, global_pool=False, stride=FLAGS.backbone_stride, scope='fuck1')
        backbone1 = backbone
        backbone2 = backbone
        #backbone2 = aardvark.create_stock_slim_network(FLAGS.backbone, self.images, self.is_training, global_pool=False, stride=FLAGS.backbone_stride, scope='fuck2')

        upscale = FLAGS.backbone_stride // FLAGS.anchor_stride
        with slim.arg_scope(aardvark.default_argscope(self.is_training)):
            
            # anchor activation signals
            #xxx = slim.conv2d(backbone, 64, 3, 1)
            xxx = backbone1
            xxx = slim.conv2d_transpose(xxx, 64, 2*upscale, upscale)
            if FLAGS.dice:
                #anchors = slim.conv2d_transpose(xxx, self.n_priors, 2*upscale, upscale, activation_fn=tf.sigmoid)
                anchors = slim.conv2d(xxx, self.n_priors, 1, 1, activation_fn=tf.sigmoid)
                dice, dice_chs = weighted_dice_loss_by_channel(self.gt_anchors, anchors, self.gt_anchors_weight, self.n_priors)
                dice = tf.identity(dice, name='di')
                anchor_layer_shape = tf.shape(anchors)
                if not FLAGS.rcnnonly:
                    tf.losses.add_loss(dice * FLAGS.di_weight1)
                    self.metrics.append(dice)
                    #for i, d in enumerate(dice_chs):
                    #    self.metrics.append(tf.identity(d, name='d%d' % i))


                prob = tf.reshape(anchors, (-1,))

            else:
                #logits = slim.conv2d_transpose(xxx, self.n_priors * 2, 2*upscale, upscale, activation_fn=tf.sigmoid)
                logits = slim.conv2d(xxx, self.n_priors * 2, 1, 1, activation_fn=None)
                anchor_layer_shape = tf.shape(logits)
                logits2 = tf.reshape(logits, (-1, 2))   # ? * 2
                gt_anchors = tf.reshape(self.gt_anchors, (-1, ))
                gt_anchors_weight = tf.reshape(self.gt_anchors_weight, (-1,))
                xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=tf.cast(gt_anchors, dtype=tf.int32))
                xe = xe * gt_anchors_weight
                xe = tf.reduce_sum(xe) / (tf.reduce_sum(gt_anchors_weight) + 0.0001)
                xe = tf.identity(xe, name='xe')

                prob = tf.squeeze(tf.slice(tf.nn.softmax(logits2), [0, 1], [-1, 1]), axis=1)
                if not FLAGS.rcnnonly:
                    tf.losses.add_loss(xe * FLAGS.di_weight1)
                    self.metrics.append(xe)
            '''
            '''

            '''
            self.metrics.append(tf.reduce_mean(anchors, name='o'))
            self.metrics.append(tf.reduce_sum(self.gt_anchors * anchors, name='s1'))
            self.metrics.append(tf.reduce_sum(self.gt_anchors, name='s2'))
            self.metrics.append(tf.reduce_sum(anchors, name='s3'))
            self.metrics.append(tf.reduce_mean(self.gt_params_weight, name='o'))
            '''



            #xxx = slim.conv2d(backbone, 128, 3, 1)
            xxx = backbone2
            xxx = slim.conv2d_transpose(xxx, 256, 2*upscale, upscale)
            #params = slim.conv2d_transpose(xxx, 4 * self.n_priors, 2*upscale, upscale, activation_fn=None)
            params = slim.conv2d(xxx, 4 * self.n_priors, 1, 1, activation_fn=None)
            params = tf.reshape(params, (-1, 4))     # ? * 4
            gt_params = tf.reshape(self.gt_params, (-1, 4))
            gt_params_weight = tf.reshape(self.gt_params_weight, (-1,))

            gt_params_weight = tf.check_numerics(gt_params_weight, 'gt_params_weight')
            pl = weighted_loss_by_channel(params_loss_rpn(params, gt_params, self.priors_wh), gt_params_weight, self.n_priors)

            pl = tf.check_numerics(pl, 'p1', name='p1') # params-loss

            if not FLAGS.rcnnonly:
                tf.losses.add_loss(pl * FLAGS.pl_weight1)
                self.metrics.append(pl)

            #prob = tf.reshape(anchors, (-1,))
            # index is index within mini batch
            boxes, index = anchors2boxes(anchor_layer_shape, params, self.priors_wh)

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

            #self.metrics.append(tf.identity(tf.cast(tf.shape(boxes)[0], dtype=tf.float32), name='o'))
            sel = tf.image.non_max_suppression(shift_boxes(boxes, index), prob, nms_max, iou_threshold=nms_th)
            # sel is a list of indices
            rpn_probs = tf.gather(prob, sel, name='rpn_probs')
            rpn_boxes = tf.gather(boxes, sel, name='rpn_boxes')
            rpn_index = tf.gather(index, sel, name='rpn_index')

            n_hits, rpn_hits, gt_hits = tf.py_func(self.gt_matcher.apply, [rpn_boxes, rpn_index, self.gt_boxes], [tf.float32, tf.int32, tf.int32])

            self.metrics.append(tf.identity(tf.cast(tf.shape(rpn_boxes)[0], dtype=tf.float32), name='n'))
            self.metrics.append(tf.identity(tf.cast(n_hits, dtype=tf.float32), name='h'))

            # % boxes found
            precision = n_hits / (tf.cast(tf.shape(rpn_boxes)[0], tf.float32) + 0.0001);
            recall = n_hits / (tf.cast(tf.shape(self.gt_boxes)[0], tf.float32) + 0.0001);
            self.metrics.append(tf.identity(precision, name='p'))
            self.metrics.append(tf.identity(recall, name='r'))

            # setup prediction
            # normalize boxes to [0-1]
            boxes = normalize_boxes(tf.shape(self.images), rpn_boxes)

            # we need to add extra samples from training boxes only

            if False:
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
                params = slim.fully_connected(net, FLAGS.classes * 4, activation_fn=None)
                params = tf.reshape(params, [-1, FLAGS.classes, 4])
            else:   # my simplified simplementation
                if FLAGS.rcnnonly:
                    backbone = tf.stop_gradient(backbone)

                mask_size = FLAGS.pooling_size * 2
                net = tf.image.crop_and_resize(backbone1, boxes, rpn_index, [mask_size, mask_size])
                net = slim.conv2d(net, 256, 3, 1)
                net = slim.conv2d(net, 256, 3, 1)
                net = slim.max_pool2d(net, [2,2], padding='SAME')
                net = slim.conv2d(net, 512, 3, 1)
                net = slim.conv2d(net, 512, 3, 1)
                #net = slim.conv2d(patches, 64, 3, 1)
                net = tf.reduce_mean(net, [1, 2], keep_dims=False)
                logits = slim.fully_connected(net, FLAGS.classes, activation_fn=None)

                #net = slim.conv2d(patches, 128, 3, 1)
                #net = patches
                #net = tf.reduce_mean(net, [1, 2], keep_dims=False)
                params = slim.fully_connected(net, FLAGS.classes * 4, activation_fn=None)
                params = tf.reshape(params, [-1, FLAGS.classes, 4])

            if FLAGS.classes == 2:
                logits = tf.clip_by_value(logits, -10, 10)

            
            tf.nn.softmax(logits, name='probs')
            cls = tf.argmax(logits, axis=1, name='cls')


            if True:    # for inference stage
                onehot = tf.expand_dims(tf.one_hot(tf.cast(cls, tf.int32), depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
                # onehot: N * C * 1
                # params : N * C * 4
                params_onehot = tf.reduce_sum(params * onehot, axis=1)
                refined_boxes = refine_bbox(rpn_boxes, params_onehot)
                tf.identity(refined_boxes, name='boxes')

            rpn_boxes = tf.gather(rpn_boxes, rpn_hits)
            logits = tf.gather(logits, rpn_hits)
            params = tf.gather(params, rpn_hits)

            '''
            self.metrics.append(tf.reduce_sum(tf.nn.l2_loss(logits), name='U'))
            self.metrics.append(tf.reduce_sum(tf.nn.l2_loss(params), name='V'))
            '''


            matched_gt_boxes = tf.gather(self.gt_boxes, gt_hits)
            matched_gt_labels = tf.cast(tf.squeeze(tf.slice(matched_gt_boxes, [0, 1], [-1, 1]), axis=1), tf.int32)
            matched_gt_boxes = transform_bbox(rpn_boxes, tf.slice(matched_gt_boxes, [0, 3], [-1, 4]))

            onehot = tf.expand_dims(tf.one_hot(matched_gt_labels, depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
            params = tf.reduce_sum(params * onehot, axis=1)

            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=matched_gt_labels)
            xe = tf.check_numerics(tf.reduce_sum(xe)/(n_hits + 0.0001), 'x2', name='x2')

            if not FLAGS.rpnonly:
                tf.losses.add_loss(xe * FLAGS.xe_weight2)
                self.metrics.append(xe)

            pl = params_loss(params, matched_gt_boxes) 
            pl = tf.reduce_sum(pl) / (n_hits + 0.0001)
            pl = tf.check_numerics(pl, 'p2', name='p2') # params-loss
            if not FLAGS.rpnonly:
                tf.losses.add_loss(pl*FLAGS.pl_weight2)
                self.metrics.append(pl)
        pass

    def extra_stream_config (self, is_training):
        if len(self.priors) > 0:
            aardvark.print_red('priors %s' % str(self.priors))

        augments = aardvark.load_augments(is_training)
        shift = 0
        if is_training:
            shift = FLAGS.clip_shift
        return {
                  "annotate": [1],
                  "transforms": 
                      [{"type": "resize", "max_size": FLAGS.max_size}] +
                      augments + [
                      #{"type": "clip", "round": FLAGS.backbone_stride},
                      {"type": "clip", "shift": shift, "width": FLAGS.fix_width, "height": FLAGS.fix_height, "round": FLAGS.clip_stride},
                      {"type": "anchors.dense.box", 'downsize': FLAGS.anchor_stride, 'lower_th': FLAGS.lower_th, 'upper_th': FLAGS.upper_th, 'weighted': False, 'priors': self.priors, 'params_default': 1.0},
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

