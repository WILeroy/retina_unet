from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def softmax_cross_entropy(labels, logits, n_classes, weights=None):
    """ Computes cross entropy between labels and logits.

    Args:
        labels: an array of numpy or a 'Tensor', shape = [?, W, H, n_classes].
        logits: output of the model, no softmax, shape = [?, W, H, n_classes].
        weights: Node, an array of numpy or a tensor, shape = [n_classes].
    
    Returns:
        The value of the weighted cross entropy.
    """

    with tf.name_scope('loss'):
        if weights is None:
            weights = tf.ones([n_classes], dtype=tf.float32)
        else:
            weights = tf.cast(weights, dtype=tf.float32)

        flat_labels = tf.cast(tf.reshape(labels, [-1, n_classes]), dtype=tf.float32)
        flat_logits = tf.cast(tf.reshape(logits, [-1, n_classes]), dtype=tf.float32)

        epsilon = tf.constant(value=1e-7)
        logits_softmax = tf.nn.softmax(flat_logits) + epsilon

        weight_cross_entropy = -tf.reduce_sum(tf.multiply(flat_labels*tf.log(logits_softmax), weights),
                                              axis=1)

    return tf.reduce_mean(weight_cross_entropy)

def softmax_focal_loss(labels, logits, n_classes, gamma=2., alpha=.25):
    """ Computes focal loss between labels and logits.
    """

    with tf.name_scope('loss'):
        flat_labels = tf.cast(tf.reshape(labels, [-1, n_classes]), dtype=tf.float32)
        flat_logits = tf.cast(tf.reshape(logits, [-1, n_classes]), dtype=tf.float32)

        epsilon = tf.constant(value=1e-7)
        logits_softmax = tf.nn.softmax(flat_logits) + epsilon

        coef = alpha * tf.pow(1-logits_softmax, gamma)
        focal_loss = -tf.reduce_sum(flat_labels*tf.log(logits_softmax)*coef, axis=1)

    return tf.reduce_mean(focal_loss)

if __name__ == '__main__':
    labels = np.array([0, 1, 1, 2])
    labels_onehot = to_categorical(labels)
    logits = np.array([0.6, 0.2, 0.2, 0.4, 0.5, 0.1, 0.3, 0.6, 0.1, 0.05, 0.05, 0.9]).reshape([4, 3])

    fl = softmax_focal_loss(labels=labels_onehot, logits=logits, n_classes=3)
    with tf.Session() as sess:
        print(sess.run(fl))