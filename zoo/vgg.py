from tensorflow import nn
from tensorflow import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected, batch_norm

# https://arxiv.org/pdf/1409.1556.pdf
# Notes:
#   - default size 224x224
#   - input should be normalized by substracting mean pixel value
#   - conv stride is always 1, down-sizing achieved by max_pool
#   - conv padding is SAME
#   - all intermediate conv2d have relu

# Tensorflow defaults
#   - conv2: SAME, relu
#   - max_pool, VALID

def classification_head (net, num_classes):
    net = flatten(net)
    net = fully_connected(net, 4096)
    net = fully_connected(net, 4096)
    net = fully_connected(net, num_classes, activation_fn=None)
    return net

F1x1 = 1
Flrn = 2

configs = {'a': [[64, 1], [128, 1], [256, 2], [512, 2], [512, 2]],
           'a_lrn': [[64, 1, Flrn], [128, 1], [256, 2], [512, 2], [512, 2]],
           'b': [[64, 2], [128, 2], [256, 2], [512, 2], [512, 2]],
           'c': [[64, 2], [128, 2], [256, 2, F1x1], [512, 2, F1x1], [512, 2, F1x1]],
           'd': [[64, 2], [128, 2], [256, 3], [512, 3], [512, 3]],
           'e': [[64, 2], [128, 2], [256, 4], [512, 4], [512, 4]],
          }

def backbone (net, config, conv2d_params):
    for block in config:
        if len(block) == 2:
            depth, times = block
            flag = 0
        else:
            depth, times, flag = block

        for _ in range(times):
            net = conv2d(net, depth, 3, **conv2d_params)
            pass
        if flag == F1x1:
            net = conv2d(net, depth, 1, **conv2d_params)
            pass
        elif flag == Flrn:
            raise Exception('LRN not implemented')
        net = max_pool2d(net, 2, 2)
    return net

def vgg (net, num_classes=None, flavor='a', scope=None, conv2d_params = {}):
    if scope is None:
        scope = 'vgg_' + flavor
    with variable_scope(scope):
        net = backbone(net, configs[flavor], conv2d_params)
        if not num_classes is None:
            net = classification_head(net, num_classes)
            pass
    return net

def vgg_bn (net, is_training, num_classes=None, flavor='a', scope=None):
    return vgg(net, num_classes, flavor, scope, 
                {"normalizer_fn": batch_norm,
                 "normalizer_params": {"is_training": is_training,
                                       "decay": 0.9,
                                       "epsilon": 5e-4}})

