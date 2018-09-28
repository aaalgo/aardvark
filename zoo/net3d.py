import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('re_weight', 0.0001, 'regularization weight')

def unet (X, is_training):
    BN = False
    net = X
    stack = []
    with tf.name_scope('myunet'):
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.re_weight)

        def conv (input, channels, filter_size=3, stride=1):
            if BN:
                input = tf.layers.conv3d(input, channels, filter_size, stride, padding='SAME', activation=None, kernel_regularizer=regularizer)
                input = tf.layers.batch_normalization(input, training=is_training)
                return tf.nn.relu(input)
            return tf.layers.conv3d(input, channels, filter_size, stride, padding='SAME', activation=tf.nn.relu, kernel_regularizer=regularizer)

        def max_pool (input, filter_size=3, stride=2):
            return tf.layers.max_pooling3d(input, filter_size, stride, padding='SAME')

        def conv_transpose (input, channels, filter_size=4, stride=2):
            if BN:
                input = tf.layers.conv3d_transpose(input, channels, filter_size, stride, padding='SAME', activation=None, kernel_regularizer=regularizer)
                input = tf.layers.batch_normalization(input, training=is_training)
                return tf.nn.relu(input)
            return tf.layers.conv3d_transpose(input, channels, filter_size, stride, padding='SAME', activation=tf.nn.relu, kernel_regularizer=regularizer)

        net = conv(net, 32)
        net = conv(net, 32)
        stack.append(net)       # 1/1
        net = conv(net, 64)
        net = conv(net, 64)
        net = max_pool(net)
        stack.append(net)       # 1/2
        net = conv(net, 128)
        net = conv(net, 128)
        net = max_pool(net)
        stack.append(net)       # 1/4
        net = conv(net, 256)
        net = conv(net, 256)
        net = max_pool(net)
                                # 1/8
        net = conv(net, 512)
        net = conv(net, 512)
        net = conv_transpose(net, 128)
                                # 1/4
        net = tf.concat([net, stack.pop()], 4)
        net = conv_transpose(net, 64)
        net = conv(net, 64)
                                # 1/2
        net = tf.concat([net, stack.pop()], 4)
        net = conv_transpose(net, 32)
        net = conv(net, 32)
                                # 1
        net = tf.concat([net, stack.pop()], 4)
        net = conv(net, 16)
        assert len(stack) == 0
    return net, 8

