import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class IdentityFlow(BaseNormalizingFlow):
    """
    Implements the identity bijector y = x
    """

    def __init__(self, params, n_dims, name='IdentityFlow'):
        """
        :param params: shape (?, 1), this will become alpha and define the slow of ReLU for x < 0
        :param n_dims: Dimension of the distribution that's being transformed
        """
        super(IdentityFlow, self).__init__(params,
                                           n_dims,
                                           name=name)

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow. For this flow it's irrelevant
        :return: (int) The dimension of the parameter space for the flow. This flow doesn't have parameters, hence it's always 0
        """
        return 0

    def _forward(self, x):
        return x

    def _inverse(self, y):
        return y

    def _forward_log_det_jacobian(self, x):
        return tf.zeros((tf.shape(x)[0], 1))

    def _ildj(self, y):
        return tf.zeros((tf.shape(y)[0], 1))
