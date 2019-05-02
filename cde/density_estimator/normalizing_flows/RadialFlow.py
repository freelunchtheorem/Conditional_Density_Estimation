import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class InvertedRadialFlow(BaseNormalizingFlow):
    """
    Implements a bijector x = y + (alpha * beta * (y - y_0)) / (alpha + abs(y - y_0)).

    Args:
        params: Tensor shape (?, n_dims+2). This will be split into the parameters
            alpha (?, 1), beta (?, 1), gamma (?, n_dims).
            Furthermore alpha will be constrained to assure the invertability of the flow
        n_dims: The dimension of the distribution that will be transformed
        name: The name to give this particular flow

    """
    _alpha = None
    _beta = None
    _gamma = None

    def __init__(self, params, n_dims, validate_args=False, name='InvertedRadialFlow'):
        """
        Parameter shapes (assuming you're transforming a distribution over d-space):

        shape alpha = (?, 1)
        shape beta = (?, 1)
        shape gamma = (?, ndims)
        """
        super(InvertedRadialFlow, self).__init__(params, n_dims, validate_args=validate_args, name=name)

        # split the input parameter into the individual parameters alpha, beta, gamma
        a_index, b_index, g_index = 0, 1, 2
        flow_params = [InvertedRadialFlow._handle_input_dimensionality(x)
                       for x in tf.split(value=params, num_or_size_splits=[1, 1, n_dims], axis=1)]

        # constraining the parameters before they are assigned to ensure invertibility
        self._alpha = InvertedRadialFlow._alpha_circ(flow_params[a_index])
        self._beta = InvertedRadialFlow._beta_circ(flow_params[b_index])
        self._gamma = flow_params[g_index]

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims:  The dimension of the distribution to be transformed by the flow
        :return: (int) The dimension of the parameter space for the flow
        """
        return 1 + 1 + n_dims

    def _r(self, z):
        return tf.reduce_sum(tf.abs(z - self._gamma), 1, keepdims=True)

    def _h(self, r):
        return 1. / (self._alpha + r)

    def _inverse(self, z):
        """
        Runs a forward pass through the bijector
        """
        z = InvertedRadialFlow._handle_input_dimensionality(z)
        r = self._r(z)
        h = self._h(r)
        return z + (self._alpha * self._beta * h) * (z - self._gamma)

    def _ildj(self, z):
        """
        Computes the ln of the absolute determinant of the jacobian
        """
        z = InvertedRadialFlow._handle_input_dimensionality(z)
        r = self._r(z)
        h = self._h(r)
        der_h = tf.gradients(h, [r])[0]
        ab = self._alpha * self._beta
        det = (1. + ab * h)**(self.n_dims - 1) * (1. + ab*h + ab*der_h*r)
        return tf.log(det)

    @staticmethod
    def _alpha_circ(alpha):
        """
        Method for constraining the alpha parameter to meet the invertibility requirements
        """
        return tf.nn.softplus(alpha)

    @staticmethod
    def _beta_circ(beta):
        """
        Method for constraining the beta parameter to meet the invertibility requirements
        """
        return tf.exp(beta) - 1.

    def forward(self, x):
        """
        We don't require sampling and it would be slow, therefore it is not implemented

        :raise NotImplementedError:
        """
        raise NotImplementedError()
