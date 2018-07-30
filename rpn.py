#!/usr/bin/env python3
import os
import math
import sys
from abc import abstractmethod
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

flags.DEFINE_string('priors', 'priors', '')
flags.DEFINE_integer('anchor_stride', 4, '')

flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_integer('nms_max', 128, '')
flags.DEFINE_float('nms_th', 0.5, '')
flags.DEFINE_float('match_th', 0.5, '')
flags.DEFINE_integer('max_masks', 128, '')

flags.DEFINE_float('lower_th', 0.1, '')
flags.DEFINE_float('upper_th', 0.5, '')

# optimizer settings
flags.DEFINE_float('rpn_act_weight', 1.0, '')
flags.DEFINE_float('rpn_params_weight', 1.0, '')

dump_cnt = 0

class BasicRPN (aardvark.Model2D):
    # RPN for generic shape
    def __init__ (self, min_size=1):
        super().__init__()
        #self.gt_matcher = cpp.GTMatcher(FLAGS.match_th, FLAGS.max_masks, min_size)
        self.priors = []
        if os.path.exists(FLAGS.priors):
            with open(FLAGS.priors, 'r') as f:
                for l in f:
                    if l[0] == '#':
                        continue
                    s, r = l.strip().split(' ')
                    s, r = float(s), float(r)
                    # w * h = s * s
                    # w / h = r
                    w = math.sqrt(s * s * r)
                    h = math.sqrt(s * s / r)
                    self.priors.append([w, h])
                    pass
                pass
            pass
        aardvark.print_red("PRIORS %s" % str(self.priors))
        # TODO: need a better way to generalize this to multiple priors and 0 priors
        self.n_priors = len(self.priors)
        if self.n_priors == 0:
            self.n_priors = 1
        pass

    @abstractmethod
    def rpn_params_size (self):
        pass

    @abstractmethod
    def rpn_params_loss (self, params, gt_params, priors):
        pass

    @abstractmethod
    def build_backbone (self, image):
        pass

    @abstractmethod
    def rpn_activation (self, channels, stride):
        pass

    @abstractmethod
    def rpn_parameters (self, channels, stride):
        pass

    @abstractmethod
    def rpn_generate_shapes (self, shape, anchor_params, priors, n_priors):
        pass

    def non_max_supression (self):
        return None

    def build_rpn (self, images, add_loss=True):
        # Set up model inputs
        # parameters

        if not hasattr(self, 'is_training'):
            self.is_training = tf.placeholder(tf.bool, name="is_training")
        anchor_th = tf.constant(FLAGS.anchor_th, dtype=tf.float32, name="anchor_th")

        # the reset are for training only
        # whether a location should be activated
        self.gt_anchors = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        # parameter of that location
        self.gt_params = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors * self.rpn_params_size()))
        self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.n_priors))
        #self.gt_boxes = tf.placeholder(tf.float32, shape=(None, 7))
        if len(self.priors) > 0:
            priors = tf.expand_dims(tf.constant(self.priors, dtype=tf.float32), axis=0)
        else:
            priors = tf.constant([[[1,1]]], dtype=tf.float32)
        # 1 * priors * 2

        priors2 = tf.tile(priors,[1,1,2])

        self.build_backbone(images)

        if FLAGS.dice:
            anchors = self.rpn_activation(self.n_priors, FLAGS.anchor_stride)
            anchors = tf.sigmoid(anchors)
            dice, dice_chs = weighted_dice_loss_by_channel(self.gt_anchors, anchors, self.gt_anchors_weight, self.n_priors)
            activation_loss = tf.identity(dice, name='di')
            prob = tf.reshape(anchors, (-1,))
        else:
            logits = self.rpn_activation(self.n_priors * 2, FLAGS.anchor_stride)
            logits2 = tf.reshape(logits, (-1, 2))   # ? * 2
            gt_anchors = tf.reshape(self.gt_anchors, (-1, ))
            gt_anchors_weight = tf.reshape(self.gt_anchors_weight, (-1,))
            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=tf.cast(gt_anchors, dtype=tf.int32))
            xe = tf.reduce_sum(xe * gt_anchors_weight) / (tf.reduce_sum(gt_anchors_weight) + 0.0001)
            activation_loss = tf.identity(xe, name='xe')

            prob = tf.squeeze(tf.slice(tf.nn.softmax(logits2), [0, 1], [-1, 1]), axis=1)
            pass

        params = self.rpn_parameters(self.rpn_params_size() * self.n_priors, FLAGS.anchor_stride)
        anchor_layer_shape = tf.shape(params)
        params = tf.reshape(params, (-1, self.n_priors, self.rpn_params_size()))     # ? * 4
        gt_params = tf.reshape(self.gt_params, (-1, self.n_priors, self.rpn_params_size()))
        gt_params_weight = tf.reshape(self.gt_params_weight, (-1, self.n_priors))

        pl = weighted_loss_by_channel(self.rpn_params_loss(params, gt_params, priors2), gt_params_weight, self.n_priors)
        pl = tf.check_numerics(pl, 'p1', name='p1') # params-loss

        tf.identity(prob, name='rpn_all_probs')

        if add_loss:
            tf.losses.add_loss(activation_loss * FLAGS.rpn_act_weight)
            self.metrics.append(activation_loss)
            tf.losses.add_loss(pl * FLAGS.rpn_params_weight)
            self.metrics.append(pl)

        #prob = tf.reshape(anchors, (-1,))
        # index is index within mini batch
        shapes, index = self.rpn_generate_shapes(anchor_layer_shape, params, priors2, self.n_priors)

        with tf.device('/cpu:0'):
            # fuck tensorflow, these lines fail on GPU

            # pre-filtering by threshold so we put less stress on non_max_suppression
            sel = tf.greater_equal(prob, anchor_th)
            # sel is a boolean mask

            # select only boxes with prob > th for nms
            prob = tf.boolean_mask(prob, sel)
            #params = tf.boolean_mask(params, sel)
            shapes = tf.boolean_mask(shapes, sel)
            index = tf.boolean_mask(index, sel)

        #self.metrics.append(tf.identity(tf.cast(tf.shape(boxes)[0], dtype=tf.float32), name='o'))
        sel = self.non_max_supression(shapes, index, prob)
        if sel is None:
            self.rpn_probs = tf.identify(prob, name='rpn_probs')
            self.rpn_shapes = tf.identify(shapes, name='rpn_shapes')
            self.rpn_index = tf.identify(index, name='rpn_index')
        else:
            self.rpn_probs = tf.gather(prob, sel, name='rpn_probs')
            self.rpn_shapes = tf.gather(shapes, sel, name='rpn_shapes')
            self.rpn_index = tf.gather(index, sel, name='rpn_index')
            pass
        
        # sel is a list of indices
        pass

def shift_boxes (boxes, offset):
    # boxes     N * [x1, y1, x2, y2]
    # offsets   N
    offset = tf.expand_dims(offset * FLAGS.max_size * 2, axis=1)
    # offset    N * [V]
    # such that there's no way for boxes from different offset to overlap
    return boxes + tf.cast(offset, dtype=tf.float32)

# box rpns

class RPN (BasicRPN):

    def __init__ (self, min_size=1):
        super().__init__()

    def rpn_params_size (self):
        return 4

    def rpn_params_loss (self, params, gt_params, priors):
        # params        ? * priors * 4
        # gt_params     ? * priors * 4
        # priors        1 * priors * 2

        gt_params = gt_params / priors

        l1 = tf.losses.huber_loss(params, gt_params, reduction=tf.losses.Reduction.NONE, loss_collection=None)
        return tf.reduce_sum(l1, axis=2)

    def rpn_generate_shapes (self, shape, anchor_params, priors, n_priors):
        # anchor parameters are: dx, dy, w, h
        # anchor_params: n * n_priors * 4
        # priors: 1 * priors * 2
        B = shape[0]
        H = shape[1]
        W = shape[2]
        offset = tf_repeat(tf.range(B), [H * W * n_priors])
        if True:    # generate array of box centers
            x0 = tf.cast(tf.range(W) * FLAGS.anchor_stride, tf.float32)
            y0 = tf.cast(tf.range(H) * FLAGS.anchor_stride, tf.float32)
            x0, y0 = tf.meshgrid(x0, y0)
            x0 = tf.reshape(x0, (-1,))
            y0 = tf.reshape(y0, (-1,))
            x0 = tf.tile(tf_repeat(x0, [n_priors]), [B])
            y0 = tf.tile(tf_repeat(y0, [n_priors]), [B])

        anchor_params = tf.reshape(anchor_params * priors, (-1, 4))
        
        dx, dy, w, h = [tf.squeeze(x, axis=1) for x in tf.split(anchor_params, [1,1,1,1], 1)]

        W = tf.cast(W * FLAGS.anchor_stride, tf.float32)
        H = tf.cast(H * FLAGS.anchor_stride, tf.float32)

        max_X = W-1
        max_Y = H-1

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

    def non_max_supression (self, boxes, index, prob):
        return tf.image.non_max_suppression(shift_boxes(boxes, index), prob, self.nms_max, iou_threshold=self.nms_th)

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        self.nms_max = tf.constant(FLAGS.nms_max, dtype=tf.int32, name="nms_max")
        self.nms_th = tf.constant(FLAGS.nms_th, dtype=tf.float32, name="nms_th")

        self.build_rpn(self.images, True)
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
                      {"type": "rasterize"},
                      ]
                 }

    def feed_dict (self, record, is_training = True):
        global dump_cnt
        _, images, _, gt_anchors, gt_anchors_weight, \
                      gt_params, gt_params_weight = record
        assert np.all(gt_anchors < 2)
        #gt_boxes = np.reshape(gt_boxes, [-1, 7])    # make sure shape is correct
        if dump_cnt < 20:
            # dump images for sanity check
            for i in range(images.shape[0]):
                cv2.imwrite('picpac_dump2/%d_a_image.png' % dump_cnt, images[i])
                for j in range(gt_anchors.shape[3]):
                    cv2.imwrite('picpac_dump2/%d_b_%d_anchor.png' % (dump_cnt, j), gt_anchors[i,:,:,j]*255)
                    cv2.imwrite('picpac_dump2/%d_c_%d_mask.png' % (dump_cnt, j), gt_anchors_weight[i,:,:,j]*255)
                    cv2.imwrite('picpac_dump2/%d_d_%d_weight.png' % (dump_cnt, j), gt_params_weight[i,:,:,j]*255)
                dump_cnt += 1

        return {self.is_training: is_training,
                self.images: images,
                self.gt_anchors: gt_anchors,
                self.gt_anchors_weight: gt_anchors_weight,
                self.gt_params: gt_params,
                self.gt_params_weight: gt_params_weight}
