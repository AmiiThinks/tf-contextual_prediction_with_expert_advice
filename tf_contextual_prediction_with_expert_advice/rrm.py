import tensorflow as tf
from .__init__ import utility


def rm_policy(regrets):
    num_actions = regrets.shape[1].value
    qregrets = tf.maximum(regrets, 0.0)
    z = tf.tile(
        tf.reduce_sum(qregrets, axis=1, keepdims=True), [1, num_actions])
    uniform_strat = tf.fill(tf.shape(qregrets), 1.0 / num_actions)
    return tf.where(tf.greater(z, 0.0), qregrets / z, uniform_strat)


def rrm_utilities(model, contexts, action_utilities):
    return utility(rm_policy(model(contexts)), action_utilities)


def rrm_loss(model, contexts, action_utilities, ignore_negative_regrets=True):
    regrets = model(contexts)
    num_actions = regrets.shape[1].value
    policy = rm_policy(regrets)

    inst_regret = tf.stop_gradient(
        action_utilities - tf.tile(
            utility(policy, action_utilities),
            [1, num_actions]
        )
    )  # yapf:disable

    regret_diffs = tf.square(regrets - inst_regret)
    if ignore_negative_regrets:
        is_substantive_regret_diff = tf.stop_gradient(
            tf.logical_or(
                tf.greater(regrets, 0.0), tf.greater(inst_regret, 0.0)))

        regret_diffs = tf.where(is_substantive_regret_diff, regret_diffs,
                                tf.zeros_like(inst_regret))

    return tf.reduce_mean(regret_diffs) / 2.0


def rrm_grad(model, contexts, action_utilities):
    with tf.GradientTape() as tape:
        loss_value = rrm_loss(model, contexts, action_utilities)
    return zip(tape.gradient(loss_value, model.variables), model.variables)
