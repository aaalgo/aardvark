import tensorflow as tf
import tensorflow.contrib.slim as slim

def dice_loss (gt, prob):
    return -2 * (tf.reduce_sum(gt * prob) + 0.00001) / (tf.reduce_sum(gt) + tf.reduce_sum(prob) + 0.00001)

def weighted_dice_loss (gt, prob, weight):
    return -2 * (tf.reduce_sum(gt * prob * weight) + 0.00001) / (tf.reduce_sum(gt * weight) + tf.reduce_sum(prob * weight) + 0.00001)

def weighted_dice_loss_by_channel (gt, prob, weight, channels):
    split = [1] * channels
    gt = tf.split(gt, split, axis=3)
    prob = tf.split(prob, split, axis=3)
    weight = tf.split(weight, split, axis=3)
    dice = []
    for i in range(channels):
        dice.append(weighted_dice_loss(gt[i], prob[i], weight[i]))
        pass
    return tf.add_n(dice) / channels, dice

def weighted_loss_by_channel (loss, weight, channels):
    #loss = tf.reshape(loss, (-1, channels))
    #weight = tf.reshape(weight, (-1, channels))
    loss = tf.reduce_sum(loss * weight, axis=0)
    loss = tf.reshape(loss, (channels,))
    weight = tf.reduce_sum(weight, axis=0) + 0.0001
    weight = tf.reshape(weight, (channels,))
    return tf.reduce_mean(loss/weight)


def tf_repeat(tensor, repeats):
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


def slim_multistep_upscale (net, octaves, reduction, step=2):
    ch = net.get_shape()[3]
    print("UPSCALE", ch, reduction)
    ch = ch // reduction
    while octaves > 1:
        ch = ch // 2
        octaves = octaves // step
        print("UPSCALE", ch, reduction)
        net = slim.conv2d_transpose(net, ch, step * 2, step, padding='SAME')
    return net

