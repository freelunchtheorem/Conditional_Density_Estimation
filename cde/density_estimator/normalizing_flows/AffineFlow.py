import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class AffineFlow(BaseNormalizingFlow):
    """
    Implements a bijector y = (a*x) + b

    The parameters have the shape
    a: (?, n_dims)
    b: (?, n_dims)
    """
    _a = None
    _b = None

    def __init__(self, params, n_dims, validate_args=False, name='AffineFlow'):
        """
        :param params: shape (?, 2*n_dims), this will be split into the two parameters a and b
        :param n_dims: Dimension of the distribution that's being transformed
        """
        super(AffineFlow, self).__init__(params, n_dims, validate_args=validate_args, name=name)

        flow_params = [AffineFlow._handle_input_dimensionality(x)
                       for x in tf.split(value=params, num_or_size_splits=[n_dims, n_dims], axis=1)]
        self._a = flow_params[0]
        self._b = flow_params[1]
        self._n_dims = n_dims

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow.
        :return: (int) The dimension of the parameter space for the flow. Here it's n_dims + n_dims, for a and b
        """
        return 2*n_dims

    def _forward(self, x):
        """
        Forward pass through the bijector. a*x + b
        """
        x = AffineFlow._handle_input_dimensionality(x)
        return tf.exp(self._a) * x + self._b

    def _inverse(self, y):
        """
        Backward pass through the bijector. (y-b) / a
        """
        y = AffineFlow._handle_input_dimensionality(y)
        return (y - self._b) * tf.exp(-self._a)

    def _forward_log_det_jacobian(self, y):
        return tf.reduce_sum(self._a, 1, keep_dims=True)
