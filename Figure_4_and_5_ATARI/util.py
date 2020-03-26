import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


def combine_flat_list(_structure, _flat_list, axis=1):
    _combined = []
    for i in range(len(_flat_list[0])):
        t = []
        for v in _flat_list:
            t.append(v[i])
        if len(t[0].get_shape()) == 0:
            cc = tf.stack(t, axis)
        else:
            cc = tf.concat(t, axis)
        _combined.append(cc)
    return nest.pack_sequence_as(_structure, _combined)


def to_bool(_t):
    return tf.cast(_t, tf.bool)


def switch_time_and_batch_dimension(_tensor):
    rank = len(_tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[1] = 1, 0
    if _tensor.dtype == tf.bool:
        _tensor = tf.cast(_tensor, tf.int64)
    res = tf.transpose(_tensor, perm, name='switch_time_and_batch_dimension')
    if _tensor.dtype == tf.bool:
        return tf.cast(res, tf.bool)
    return res


def exp_convolve(tensor, decay, initializer=None):
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        if initializer is None:
            initializer = tf.zeros_like(tensor)

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1-decay) * x, tensor, initializer=initializer)
    return filtered_tensor
