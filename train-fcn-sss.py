#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'zoo/sss'))
import tensorflow as tf
import aardvark
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('net', 'MobileUNet', 'architecture')

class Model (aardvark.SegmentationModel):
    def __init__ (self):
        super().__init__()
        pass

    def init_session (self, sess):
        if not self.init_fn is None:
            self.init_fn(sess)
        pass

    def inference (self, net_input, num_classes, is_training):
        network = None
        init_fn = None
        if FLAGS.net == "FC-DenseNet56" or FLAGS.net == "FC-DenseNet67" or FLAGS.net == "FC-DenseNet103":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network = build_fc_densenet(net_input, preset_model = FLAGS.net, num_classes=num_classes)
        elif FLAGS.net == "RefineNet-Res50" or FLAGS.net == "RefineNet-Res101" or FLAGS.net == "RefineNet-Res152":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
            # RefineNet requires pre-trained ResNet weights
                network, init_fn = build_refinenet(net_input, preset_model = FLAGS.net, num_classes=num_classes, is_training=is_training)
        elif FLAGS.net == "FRRN-A" or FLAGS.net == "FRRN-B":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network = build_frrn(net_input, preset_model = FLAGS.net, num_classes=num_classes)
        elif FLAGS.net == "Encoder-Decoder" or FLAGS.net == "Encoder-Decoder-Skip":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network = build_encoder_decoder(net_input, preset_model = FLAGS.net, num_classes=num_classes)
        elif FLAGS.net == "MobileUNet" or FLAGS.net == "MobileUNet-Skip":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network = build_mobile_unet(net_input, preset_model = FLAGS.net, num_classes=num_classes)
        elif FLAGS.net == "PSPNet-Res50" or FLAGS.net == "PSPNet-Res101" or FLAGS.net == "PSPNet-Res152":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
            # Image size is required for PSPNet
            # PSPNet requires pre-trained ResNet weights
                network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width], preset_model = FLAGS.net, num_classes=num_classes, is_training=is_training)
        elif FLAGS.net == "GCN-Res50" or FLAGS.net == "GCN-Res101" or FLAGS.net == "GCN-Res152":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
            # GCN requires pre-trained ResNet weights
                network, init_fn = build_gcn(net_input, preset_model = FLAGS.net, num_classes=num_classes, is_training=is_training)
        elif FLAGS.net == "DeepLabV3-Res50" or FLAGS.net == "DeepLabV3-Res101" or FLAGS.net == "DeepLabV3-Res152":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
            # DeepLabV requires pre-trained ResNet weights
                network, init_fn = build_deeplabv3(net_input, preset_model = FLAGS.net, num_classes=num_classes, is_training=is_training)
        elif FLAGS.net == "DeepLabV3_plus-Res50" or FLAGS.net == "DeepLabV3_plus-Res101" or FLAGS.net == "DeepLabV3_plus-Res152":
            # DeepLabV3+ requires pre-trained ResNet weights
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network, init_fn = build_deeplabv3_plus(net_input, preset_model = FLAGS.net, num_classes=num_classes, is_training=is_training)
        elif FLAGS.net == "AdapNet":
            with slim.arg_scope(aardvark.default_argscope(is_training)):
                network = build_adaptnet(net_input, num_classes=num_classes)
        else:
            raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

        self.init_fn = init_fn
        return network

def main (_):
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

