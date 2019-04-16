import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class IdentityFlow(BaseNormalizingFlow):
    """
    Implements the identity bijector y = x
    """

    def __init__(self, params, n_dims, validate_args=False, name='IdentityFlow'):
        """
        :param params: shape (?, 1), this will become alpha and define the slow of ReLU for x < 0
        :param n_dims: Dimension of the distribution that's being transformed
        """
        super(IdentityFlow, self).__init__(params, n_dims, validate_args=validate_args, name=name)
        self._n_dims = n_dims

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow. For this flow it's irrelevant
        :return: (int) The dimension of the parameter space for the flow. Here, it's always 0
        """
        return 0

    def _forward(self, x):
        """
        Forward pass through the bijector
        """
        return x

    def _inverse(self, y):
        """
        Backward pass through the bijector
        """
        return y

    def _inverse_log_det_jacobian(self, y):
        return 0.
