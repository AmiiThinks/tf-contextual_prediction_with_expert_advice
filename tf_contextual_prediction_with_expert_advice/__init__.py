import tensorflow as tf


def utility(policy, action_utilities):
    return tf.reduce_sum(policy * action_utilities, axis=1, keepdims=True)


def normalized(v, axis=0):
    v = tf.convert_to_tensor(v)
    n = tf.shape(v)[axis]

    tile_dims = [tf.ones([], dtype=tf.int32)] * len(v.shape)
    tile_dims[axis] = n

    z = tf.tile(tf.reduce_sum(v, axis=axis, keepdims=True), tile_dims)
    ur = tf.fill(v.shape, 1.0 / tf.cast(n, tf.float32))
    return tf.where(tf.greater(z, 0.0), v / z, ur)


def l1_projection_to_simplex(v, axis=0):
    return normalized(tf.maximum(0.0, v), axis=axis)


def indmax(tensor, axis=0, tolerance=1e-15):
    almost_max = tf.reduce_max(tensor, axis=axis, keepdims=True) - tolerance
    return normalized(
        tf.where(
            tf.greater_equal(tensor, almost_max), tf.ones_like(tensor),
            tf.zeros_like(tensor)),
        axis=axis)


def br(action_utilities):
    return indmax(action_utilities, axis=1)


def behavioral_to_sequence_form_strat(policy):
    return tf.reshape(policy, [tf.size(policy), 1])


def norm_exp(x, *args, temp=1.0, **kwargs):
    x /= temp
    return tf.nn.softmax(x - tf.reduce_max(x, axis=1, keepdims=True))


def rm_policy(regrets):
    return l1_projection_to_simplex(regrets, axis=1)
