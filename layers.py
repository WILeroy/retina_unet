from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def weight_variable(shape, stddev=0.1, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W, b, keep_prob):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob)

def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        return tf.concat([x1, x2], 3)
