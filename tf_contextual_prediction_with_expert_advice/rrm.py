import tensorflow as tf
from tf_contextual_prediction_with_expert_advice import \
    utility, \
    rm_policy
from tf_contextual_prediction_with_expert_advice.policy_model import \
    PolicyModel, \
    RmPolicyActivation
from tf_contextual_prediction_with_expert_advice.learner import Learner


def rrm_utilities(model, contexts, action_utilities):
    return utility(rm_policy(model(contexts)), action_utilities)


def rrm_loss_given_policy(regrets,
                          policy,
                          action_utilities,
                          ignore_negative_regrets=True):
    num_actions = regrets.shape[1].value
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

    return tf.reduce_mean(tf.reduce_sum(regret_diffs, axis=1)) / 2.0


def rrm_loss(regrets, action_utilities, ignore_negative_regrets=True):
    regrets = tf.convert_to_tensor(regrets)
    return rrm_loss_given_policy(
        regrets,
        rm_policy(regrets),
        action_utilities,
        ignore_negative_regrets=ignore_negative_regrets)


def rrm_grad(model, contexts, action_utilities, ignore_negative_regrets=True):
    with tf.GradientTape() as tape:
        loss_value = rrm_loss(
            model(contexts),
            action_utilities,
            ignore_negative_regrets=ignore_negative_regrets)
    return zip(tape.gradient(loss_value, model.variables), model.variables)


class RrmPolicyModel(RmPolicyActivation, PolicyModel):
    pass


class RrmLearner(Learner):
    def __init__(self, *args, ignore_negative_regrets=False, **kwargs):
        super(RrmLearner, self).__init__(*args, **kwargs)
        self.ignore_negative_regrets = ignore_negative_regrets

    def loss(self, utility, inputs=None, predictions=None, policy=None):
        if predictions is None:
            predictions = self.pre_activations(inputs)
        if policy is None:
            return rrm_loss(
                predictions,
                utility,
                ignore_negative_regrets=self.ignore_negative_regrets)
        else:
            return rrm_loss_given_policy(
                predictions,
                policy,
                utility,
                ignore_negative_regrets=self.ignore_negative_regrets)
