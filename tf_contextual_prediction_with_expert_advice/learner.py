from tf_contextual_prediction_with_expert_advice.policy_model import \
    PolicyModelMixin


class Learner(PolicyModelMixin):
    def __init__(self, policy_model, optimizer):
        self.policy_model = policy_model
        self.optimizer = optimizer

    def pre_activations(self, inputs):
        return self.policy_model.pre_activations(inputs)

    def policy(self, inputs):
        return self.policy_model.policy(inputs)

    def loss(self, utility, inputs=None, predictions=None, policy=None):
        raise RuntimeError('Unimplemented')

    def apply_gradients(self, loss, tape):
        vars = self.policy_model.model.variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, vars), vars))
        return self
