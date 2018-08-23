import tensorflow as tf
from tf_contextual_prediction_with_expert_advice import norm_exp, rm_policy


class PolicyModelMixin(object):
    def pre_activations(self, input):
        return self.model(input)

    def policy(self, inputs):
        return self.policy_activation(self.pre_activations(inputs))

    def policy_activation(self, pre_activations):
        return pre_activations

    def __call__(self, inputs):
        return self.policy(inputs)


class PolicyModel(PolicyModelMixin):
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


class NormExpPolicyActivation(object):
    def policy_activation(self, pre_activations):
        return norm_exp(pre_activations)


class RmPolicyActivation(object):
    def policy_activation(self, pre_activations):
        return rm_policy(pre_activations)


class MetaPolicyModel(PolicyModelMixin):
    '''Uses a meta-policy to select between multiple policies generated
       from common model predictions.
    '''

    def __init__(self, model):
        self.model = model

    def all_policy_activations(self, pre_activations):
        raise RuntimeError('Unimplemented')

    def all_policies(self, inputs):
        return self._all_policies(self.model(inputs))

    def _all_policies(self, predictions):
        return tf.stack(self.all_policy_activations(predictions), axis=-1)

    def policy_activation(self, predictions):
        policies = self._all_policies(predictions)
        meta_policy = tf.reshape(self.meta_policy(),
                                 [1, 1, self.num_policies()])
        return tf.reduce_sum(policies * meta_policy, axis=-1)
