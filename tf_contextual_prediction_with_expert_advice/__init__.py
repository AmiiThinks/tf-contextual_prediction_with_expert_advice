import tensorflow as tf


def utility(policy, action_utilities):
    return tf.reduce_sum(policy * action_utilities, axis=1, keepdims=True)


def indmax(t):
    t = tf.convert_to_tensor(t)
    idx = tf.argmax(t, axis=1)
    return tf.one_hot(idx, t.shape[1].value)


def br(action_utilities):
    return indmax(action_utilities)


def behavioral_to_sequence_form_strat(policy):
    return tf.reshape(policy, [tf.size(policy), 1])


def norm_exp(x, *args, temp=1.0, **kwargs):
    x /= temp
    return tf.nn.softmax(x - tf.reduce_max(x, axis=1, keepdims=True))
