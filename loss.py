from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def categorical_cross_entropy(n_classes, weights=None):

    if weights is None:
        weights = tf.ones([n_classes], dtype=tf.float32)
    else:
        weights = tf.cast(weights, dtype=tf.float32)

    def _categorical_cross_entropy(labels, logits):
        """ Computes cross entropy between labels and logits.
        Args:
            labels: an array of numpy or a 'Tensor', shape = [?, W, H, n_classes].
            logits: output of the model by softmax, shape = [?, W, H, n_classes].
            weights: None, an array of numpy or a 'Tensor', shape = [n_classes].

        Returns:
            The value of the weighted cross entropy.
        """

        flat_labels = tf.cast(tf.reshape(labels, [-1, n_classes]), dtype=tf.float32)
        flat_logits = tf.cast(tf.reshape(logits, [-1, n_classes]), dtype=tf.float32)

        weight_cross_entropy = -tf.reduce_sum(
            tf.multiply(flat_labels*tf.math.log(flat_logits), weights),
            axis=1)

        return tf.reduce_mean(weight_cross_entropy)

    return _categorical_cross_entropy


if __name__ == '__main__':
    labels = np.array([0, 1, 2], dtype=np.float32)
    labels_onehot = tf.keras.utils.to_categorical(labels)
    logits = np.array([2.9, .1, .05, .3, 1.9, .1, .05, .05, 1.9],
                      dtype=np.float32).reshape(-1, 3)
    logits_softmax = tf.keras.layers.Softmax()(logits)
    
    print('labels(onehot)\n', labels_onehot)
    print('logits(softmax)\n', logits_softmax)

    print(categorical_cross_entropy(3)(labels_onehot, logits_softmax))
    print(tf.keras.losses.CategoricalCrossentropy()(
        labels_onehot, logits_softmax))
    print(tf.keras.losses.categorical_crossentropy(
        labels_onehot, logits_softmax))