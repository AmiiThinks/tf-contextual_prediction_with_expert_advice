import tensorflow as tf


def utility(policy, action_utilities):
    return tf.reduce_sum(policy * action_utilities, axis=1, keepdims=True)


def normalized(v, axis=0):
    z = tf.reduce_sum(v, axis=axis, keepdims=True)
    zero_z = tf.cast(tf.equal(z, tf.zeros_like(z)), tf.float32)
    v = v + zero_z
    z = z + tf.cast(tf.shape(v)[axis], tf.float32) * zero_z
    return v / z


def l1_projection_to_simplex(v, axis=0):
    return normalized(tf.maximum(0.0, v), axis=axis)


def indmax(tensor, axis=0, tolerance=1e-15):
    almost_max = tf.reduce_max(tensor, axis=axis, keepdims=True) - tolerance
    return normalized(
        tf.where(
            tf.greater_equal(tensor, almost_max), tf.ones_like(tensor),
            tf.zeros_like(tensor)),
        axis=axis)


def greedy_policy(action_utilities):
    return indmax(action_utilities, axis=1)


def behavioral_to_sequence_form_strat(policy):
    return tf.reshape(policy, [tf.size(policy), 1])


def norm_exp(x, *args, temp=1.0, **kwargs):
    x /= temp
    return tf.nn.softmax(x - tf.reduce_max(x, axis=1, keepdims=True))


def rm_policy(regrets):
    return l1_projection_to_simplex(regrets, axis=1)
