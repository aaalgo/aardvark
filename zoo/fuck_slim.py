import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils, resnet_v2

def patch_resnet_arg_scope (is_training):

    def resnet_arg_scope (weight_decay=0.0001):
      print('\033[91m' + 'Using patched resnet arg scope' + '\033[0m')

      batch_norm_decay=0.9
      batch_norm_epsilon=5e-4
      batch_norm_scale=False
      activation_fn=tf.nn.relu
      use_batch_norm=True

      batch_norm_params = {
          'decay': batch_norm_decay,
          'epsilon': batch_norm_epsilon,
          'scale': batch_norm_scale,
          'updates_collections': tf.GraphKeys.UPDATE_OPS,
          # don't know what it does, but seems improves cifar10 a bit
          #'fused': None,  # Use fused batch norm if possible.
          'is_training': is_training
      }
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          #Removing following 2 improves cifar10 performance
          #weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=activation_fn,
          normalizer_fn=slim.batch_norm if use_batch_norm else None,
          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          with slim.arg_scope([slim.max_pool2d], padding='SAME'):
            with slim.arg_scope([slim.dropout], is_training=is_training) as arg_sc:
                return arg_sc
    return resnet_arg_scope

def patch (is_training):
    asc = patch_resnet_arg_scope(is_training)
    keys = [key for key in nets_factory.arg_scopes_map.keys() if 'resnet_' in key or 'densenet' in key]
    for key in keys:
        nets_factory.arg_scopes_map[key] = asc

def resnet_v2_14_nmist (inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 include_root_block=False,
                 spatial_squeeze=True,
                 scope='resnet_v2_14_nist',
                 reduction=2):
  resnet_v2_block = resnet_v2.resnet_v2_block
  blocks = [
      resnet_v2_block('block1', base_depth=64//reduction, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=128//reduction, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=256//reduction, num_units=2, stride=1),
  ]
  return resnet_v2.resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=include_root_block,
      spatial_squeeze=spatial_squeeze,
      reuse=reuse,
      scope=scope)

def resnet_v2_18 (inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 include_root_block=True,
                 spatial_squeeze=True,
                 scope='resnet_v2_18',
                 reduction=1):
  resnet_v2_block = resnet_v2.resnet_v2_block
  blocks = [
      resnet_v2_block('block1', base_depth=64//reduction, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=128//reduction, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=256//reduction, num_units=2, stride=2),
      resnet_v2_block('block4', base_depth=512//reduction, num_units=2, stride=1),
  ]
  return resnet_v2.resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=include_root_block,
      spatial_squeeze=spatial_squeeze,
      reuse=reuse,
      scope=scope)

def resnet_v2_18_cifar (inputs, num_classes=None, is_training=True, global_pool=False, output_stride=None,
                        reuse=None, scope='resnet_v2_18_cifar', spatial_squeeze=True):
    #assert global_pool
    return resnet_v2_18(inputs, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, reuse=reuse, include_root_block=False, scope=scope, spatial_squeeze=spatial_squeeze)

def resnet_v2_18_slim (inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                        reuse=None, scope='resnet_v2_18_slim', spatial_squeeze=True):
    return resnet_v2_18(inputs, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, reuse=reuse, include_root_block=True, scope=scope, reduction=2, spatial_squeeze=spatial_squeeze)

def resnet_v2_50_slim(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  resnet_v2_block = resnet_v2.resnet_v2_block
  reduction=2
  blocks = [
      resnet_v2_block('block1', base_depth=64//reduction, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128//reduction, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256//reduction, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512//reduction, num_units=3, stride=1),
  ]
  return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)

def extend ():
    nets_factory.networks_map['resnet_v2_14_nmist'] = resnet_v2_14_nmist
    nets_factory.networks_map['resnet_v2_18'] = resnet_v2_18
    nets_factory.networks_map['resnet_v2_18_cifar'] = resnet_v2_18_cifar
    nets_factory.networks_map['resnet_v2_18_slim'] = resnet_v2_18_slim
    nets_factory.networks_map['resnet_v2_50_slim'] = resnet_v2_50_slim
    nets_factory.arg_scopes_map['resnet_v2_14_nmist'] = resnet_v2.resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_18'] = resnet_v2.resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_18_cifar'] = resnet_v2.resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_18_slim'] = resnet_v2.resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_50_slim'] = resnet_v2.resnet_arg_scope
    pass

