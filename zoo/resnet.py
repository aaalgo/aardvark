from tensorflow import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected, batch_norm

# https://arxiv.org/pdf/1605.07146.pdf

# https://arxiv.org/abs/1603.05027

def original_conv2d (net, depth, filter_size):
    return conv2d(net, depth, filter_size, normalizer_fn=batch_norm)

def rewired_conv2d (net, depth, filter_size):
    # https://arxiv.org/abs/1603.05027
    # original:  conv-BN-ReLU,
    # changed here to:  BN-ReLU-conv
    net = batch_norm(net)
    net = tf.nn.relu(net)
    net = tf.conv2d(net, depth, filter_size, normalizer_fn=None, activation_fn=None)
    return net

myconv2d = rewired_conv2d

def block_basic (net):
    depth = tf.shape(net)[3]
    branch = net
    branch = myconv2d(branch, depth, 3)
    branch = myconv2d(branch, depth, 3)
    return net + branch

def block_bottleneck (net):
    depth = tf.shape(net)[3]
    branch = net
    branch = myconv2d(branch, depth, 1, normalizer_fn=batch_norm)
    branch = myconv2d(branch, depth, 3, normalizer_fn=batch_norm)
    branch = myconv2d(branch, depth, 1, normalizer_fn=batch_norm)
    return net + branch


