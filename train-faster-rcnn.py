#!/usr/bin/env python3
import os
import sys
# C++ code, python3 setup.py build
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/slim'))
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import aardvark
import cpp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('classes', 2, '')

flags.DEFINE_integer('size', None, '') 
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('anchor_stride', 4, '')
flags.DEFINE_integer('features', 128, '')
flags.DEFINE_integer('rpn_channels', 128, '')
flags.DEFINE_integer('pooling_size', 7, '')
flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_integer('nms_max', 128, '')
flags.DEFINE_float('nms_th', 0.5, '')
flags.DEFINE_float('match_th', 0.5, '')
flags.DEFINE_integer('max_masks', 128, '')

flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_string('finetune', None, '')

# optimizer settings
flags.DEFINE_float('pl_weight1', 1.0/50, '')
flags.DEFINE_float('pl_weight2', 1.0/50, '')
flags.DEFINE_float('re_weight', 0.1, '')

PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])   # VGG PIXEL MEANS USED BY TF


def tf_repeat(tensor, repeats):
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

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
    l1 = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE)
    l2 = tf.losses.huber_loss(wh, wh_gt, reduction=tf.losses.Reduction.NONE)
    return tf.reduce_sum(l1+l2, axis=1)

class FasterRCNN (aardvark.Model):
    def __init__ (self, priors=1):
        super().__init__()
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
                self.anchor_th: FLAGS.anchor_th,
                self.nms_max: FLAGS.nms_max,
                self.nms_th: FLAGS.nms_th,
                self.images: images,
                self.gt_anchors: gt_anchors,
                self.gt_anchors_weight: gt_anchors_weight,
                self.gt_params: gt_params,
                self.gt_params_weight: gt_params_weight,
                self.gt_boxes: gt_boxes}

    def build_graph (self):
        if True:    # setup inputs
            # parameters
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.anchor_th = tf.placeholder(tf.float32, shape=(), name="anchor_th")
            self.nms_max = tf.placeholder(tf.int32, shape=(), name="nms_max")
            self.nms_th = tf.placeholder(tf.float32, shape=(), name="nms_th")
            # input images
            self.images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
            # the reset are for training only
            self.gt_anchors = tf.placeholder(tf.int32, shape=(None, None, None, self.priors))
            self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.priors))
            self.gt_params = tf.placeholder(tf.float32, shape=(None, None, None, self.priors * 4))
            self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, None, None, self.priors))
            self.gt_boxes = tf.placeholder(tf.float32, shape=(None, 7))

            #self.losses = []
            #self.metrics = []

        if True:    # setup backbone
            global PIXEL_MEANS
            if not FLAGS.finetune:
                patch_arg_scopes()
            network_fn = nets_factory.get_network_fn(FLAGS.backbone, num_classes=None,
                        weight_decay=FLAGS.weight_decay, is_training=self.is_training)

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'), \
                 slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':self.is_training}), \
                 slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                net, _ = network_fn(self.images-PIXEL_MEANS, global_pool=False, output_stride=FLAGS.backbone_stride)
                assert FLAGS.backbone_stride % FLAGS.anchor_stride == 0
                ss = FLAGS.backbone_stride // FLAGS.anchor_stride
                self.backbone = slim.conv2d_transpose(net, FLAGS.features, ss*2, ss)
                pass

        if True:    # setup RPN
            rpn = slim.conv2d(self.backbone, FLAGS.rpn_channels, 3, 1, padding='SAME')

            logits = slim.conv2d(rpn, 2 * self.priors, 1, 1, activation_fn=None) 
            logits = tf.reshape(logits, (-1, 2))


            gt_anchors = tf.reshape(self.gt_anchors, (-1, ))
            gt_anchors_weight = tf.reshape(self.gt_anchors_weight, (-1,))

            logits = tf.check_numerics(logits, 'logits')
            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gt_anchors)
            xe = xe * gt_anchors_weight
            xe = tf.reduce_sum(xe) / (tf.reduce_sum(gt_anchors_weight) + 1)
            xe = tf.check_numerics(xe, 'x1', name='x1')    # rpn xe

            tf.losses.add_loss(xe)
            self.metrics.append(xe)

            params = slim.conv2d(rpn, 4 * self.priors, 1, 1, activation_fn=None)
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

            prob = tf.squeeze(tf.slice(tf.nn.softmax(logits), [0, 1], [-1, 1]), 1)
            boxes, offset = anchors2boxes(tf.shape(rpn), params, self.priors)

            with tf.device('/cpu:0'):
                # fuck tensorflow, these lines fail on GPU

                # pre-filtering by threshold so we put less stress on non_max_suppression
                sel = tf.greater_equal(prob, self.anchor_th)
                # sel is a boolean mask

                # select only boxes with prob > th for nms
                prob = tf.boolean_mask(prob, sel)
                #params = tf.boolean_mask(params, sel)
                boxes = tf.boolean_mask(boxes, sel)
                # offset is offset within minibatch
                offset = tf.boolean_mask(offset, sel)

            sel = tf.image.non_max_suppression(shift_boxes(boxes, offset), prob, self.nms_max, iou_threshold=self.nms_th)
            # sel is a list of indices
            prob = tf.gather(prob, sel)
            boxes = tf.gather(boxes, sel)
            offset = tf.gather(offset, sel)

            self.rpn_prob = tf.identity(prob, name='rpn_prob')
            self.rpn_boxes = tf.identity(boxes, name='rpn_boxes')
            self.offset = tf.identity(offset, name='offset')

            hit, self.rpn_hits, self.gt_hits = tf.py_func(self.gt_matcher.apply, [boxes, offset, self.gt_boxes], [tf.float32, tf.int32, tf.int32])

            # % boxes found
            precision = hit / (tf.cast(tf.shape(boxes)[0], tf.float32) + 0.001);
            recall = hit / (tf.cast(tf.shape(self.gt_boxes)[0], tf.float32) + 0.001);
            self.metrics.append(tf.identity(precision, name='p'))
            self.metrics.append(tf.identity(recall, name='r'))

        if True:    # setup prediction
            # normalize boxes to [0-1]
            boxes = normalize_boxes(tf.shape(self.images), self.rpn_boxes)

            mask_size = FLAGS.pooling_size * 2
            net = tf.image.crop_and_resize(self.backbone, boxes, offset, [mask_size, mask_size])
            net = slim.max_pool2d(net, [2,2], padding='SAME')
            #
            net = tf.reshape(net, [-1, FLAGS.pooling_size * FLAGS.pooling_size * FLAGS.features])

            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training)

            logits = slim.fully_connected(net, FLAGS.classes, activation_fn=None)
            prob = tf.nn.softmax(logits)
            # class probabilities
            tf.identity(prob, name='prob')

            cls = tf.argmax(logits, axis=1)
            # class prediction
            tf.identity(cls, name='cls')

            params = slim.fully_connected(net, FLAGS.classes * 4, activation_fn=None)
            params = tf.reshape(params, [-1, FLAGS.classes, 4])

            if True:    # for inference stage
                tf.identity(params, name='params')
                onehot = tf.expand_dims(tf.one_hot(tf.cast(cls, tf.int32), depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
                params1 = tf.reduce_sum(params * onehot, axis=1)
                boxes = refine_bbox(self.rpn_boxes, params1)
                tf.identity(boxes, name='boxes')

            rpn_boxes = tf.gather(self.rpn_boxes, self.rpn_hits)
            logits = tf.gather(logits, self.rpn_hits)
            params = tf.gather(params, self.rpn_hits)

            matched_gt_boxes = tf.gather(self.gt_boxes, self.gt_hits)
            matched_gt_labels = tf.cast(tf.squeeze(tf.slice(matched_gt_boxes, [0, 1], [-1, 1]), axis=1), tf.int32)
            matched_gt_boxes = transform_bbox(rpn_boxes, tf.slice(matched_gt_boxes, [0, 3], [-1, 4]))

            onehot = tf.expand_dims(tf.one_hot(matched_gt_labels, depth=FLAGS.classes, on_value=1.0, off_value=0.0), axis=2)
            params = tf.reduce_sum(params * onehot, axis=1)

            n = tf.cast(tf.shape(matched_gt_boxes)[0], tf.float32);

            '''
            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=matched_gt_labels)
            xe = tf.check_numerics(tf.reduce_sum(xe)/(n + 1), 'x2', name='x2')
            self.losses.append(xe)
            self.metrics.append(xe)

            pl = params_loss(params, matched_gt_boxes) 
            pl = tf.reduce_sum(pl) / (n + 1)
            pl = tf.check_numerics(pl * FLAGS.pl_weight2, 'p2', name='p2') # params-loss
            self.losses.append(pl)
            self.metrics.append(pl)
            '''
        pass

    def extra_stream_config (self, is_training):
        augments = aardvark.load_augments(is_training)
        return {
                  "annotate": [1],
                  "transforms": augments + [
                      {"type": "resize", "max_size": FLAGS.max_size},
                      {"type": "clip", "round": FLAGS.backbone_stride},
                      {"type": "anchors.dense.box", 'downsize': FLAGS.anchor_stride},
                      {"type": "box_feature"},
                      {"type": "rasterize"},
                      ]
                 }

def patch_arg_scopes ():
    def resnet_arg_scope (weight_decay=0.0001):
        aardvark.print_red("Patching resnet_v2 arg_scope when training from scratch")
        return resnet_utils.resnet_arg_scope(weight_decay=weight_decay,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=5e-4,
                    batch_norm_scale=False)
    nets_factory.arg_scopes_map['resnet_v1_50'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_101'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_152'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_200'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_50'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_101'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_152'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_200'] = resnet_arg_scope
    pass

def setup_finetune (ckpt, exclusions):
    print("Finetuning %s" % ckpt)
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                print("Excluding %s" % var.op.name)
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)

    variables_to_train = []
    for scope in exclusions:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    print("Training %d out of %d variables" % (len(variables_to_train), len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
    if len(variables_to_train) < 10:
        for var in variables_to_train:
            print("    %s" % var.op.name)

    return slim.assign_from_checkpoint_fn(
            ckpt, variables_to_restore,
            ignore_missing_vars=False), variables_to_train

def main (_):
    model = FasterRCNN()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

