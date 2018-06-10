from tensorflow import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers import conv2d, max_pool2d, avg_pool2d, flatten, fully_connected, batch_norm

# https://arxiv.org/pdf/1605.07146.pdf

# https://arxiv.org/abs/1603.05027

def original_conv2d (net, depth, filter_size, step = 1):
    return conv2d(net, depth, filter_size, step, normalizer_fn=batch_norm)

def rewired_conv2d (net, depth, filter_size):
    # https://arxiv.org/abs/1603.05027
    # original:  conv-BN-ReLU,
    # changed here to:  BN-ReLU-conv
    net = batch_norm(net)
    net = tf.nn.relu(net)
    net = tf.conv2d(net, depth, filter_size, normalizer_fn=None, activation_fn=None)
    return net

myconv2d = rewired_conv2d

def block (net, config, n, depth, pool):
    branch = net
    if pool:
        net = max_pool2d(net, 2, 2)
    for _ in range(n):
        for fs in config:
            if pool:
                step = 2
                pool = False
            else:
                step = 1
            branch = myconv2d(branch, depth, fs, step, normalizer_fn=batch_norm)
    return net + branch


def wrn (net, k, n, num_classes=None):
    net = block(net, [3], n, 16, False)       # 32
    net = block(net, [3,3], n, 16*k, False)   # 32
    net = block(net, [3,3], n, 32*k, True)    # 16
    net = block(net, [3,3], n, 64*k, True)    # 8
    if not num_classes is None:
        net = avg_pool2d(net, 8, 8)
        net = conv2d(net, num_classes, 1, activation_fn=None)
        pass
    return net

