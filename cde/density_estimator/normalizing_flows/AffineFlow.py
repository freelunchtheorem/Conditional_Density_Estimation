import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class AffineFlow(BaseNormalizingFlow):
    """
    Implements a bijector y = a*x + b

    Args:
        params: tensor of shape (?, 2*n_dims). This will be split into the parameters a, b
        n_dims: The dimension of the distribution that is being transformed
        name: The name to give this flow
    """
    _a = None
    _b = None

    def __init__(self, params, n_dims, name='AffineFlow'):
        super(AffineFlow, self).__init__(params,
                                         n_dims,
                                         name=name)

        flow_params = [AffineFlow._handle_input_dimensionality(x)
                       for x in tf.split(value=params, num_or_size_splits=[n_dims, n_dims], axis=1)]
        self._a = flow_params[0]
        self._b = flow_params[1]

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow.
        :return: (int) The dimension of the parameter space for the flow. Here it's n_dims + n_dims
        """
        return 2 * n_dims

    def _forward(self, x):
        """
        Forward pass through the bijector. a*x + b
        """
        return tf.exp(self._a) * x + self._b

    def _inverse(self, y):
        """
        Backward pass through the bijector. (y-b) / a
        """
        return (y - self._b) * tf.exp(-self._a)

    def _ildj(self, y):
        return -tf.reduce_sum(self._a, 1, keep_dims=True)
