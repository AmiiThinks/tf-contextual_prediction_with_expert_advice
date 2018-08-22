import tensorflow as tf
from tf_contextual_prediction_with_expert_advice import norm_exp
from tf_contextual_prediction_with_expert_advice.rrm import rm_policy


class PolicyModel(object):
    @classmethod
    def load(cls, name):
        with open('{}.yml'.format(name), "r") as yaml_file:
            data_string = yaml_file.read()
        model = tf.keras.models.model_from_yaml(data_string)
        model.load_weights('{}.h5'.format(name))
        return cls(model)

    def __init__(self, model):
        self.model = model

    def save(self, name):
        with open('{}.yml'.format(name), "w") as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights('{}.h5'.format(name))

    def policy(self, inputs):
        return self.policy_activation(self.model(inputs))

    def policy_activation(self, pre_activations):
        return pre_activations

    def __call__(self, inputs):
        return self.policy(inputs)


class NormExpPolicyActivation(object):
    def policy_activation(self, pre_activations):
        return norm_exp(pre_activations)


class RmPolicyActivation(object):
    def policy_activation(self, pre_activations):
        return rm_policy(pre_activations)


class MetaPolicyModel(PolicyModel):
    '''Uses a meta-policy to select between multiple policies generated
       from common model predictions.
    '''

    def __init__(self, meta_policy, policy_activations, *args, **kwargs):
        super(MetaPolicyModel, self).__init__(*args, **kwargs)
        self.policy_activations = policy_activations
        self.meta_policy = meta_policy

    def num_policies(self):
        return len(self.policy_activations)

    def all_policies(self, inputs):
        return self._all_policies(self.model(inputs))

    def _all_policies(self, predictions):
        return tf.stack(
            [f(predictions) for f in self.policy_activations], axis=-1)

    def policy_activation(self, predictions):
        policies = self._all_policies(predictions)
        meta_policy = tf.reshape(self.meta_policy(),
                                 [1, 1, self.num_policies()])
        return tf.reduce_sum(policies * meta_policy, axis=-1)
