import tensorflow as tf
from .BaseNormalizingFlow import BaseNormalizingFlow


class InvertedPlanarFlow(BaseNormalizingFlow):
    """
    Implements a bijector x = y + u * tanh(w_t * y + b)

    The orginal planar flows proposed in Rezende, Mohamed 2015 are designed for efficient drawing of samples
    For our CDE usecase, the inverse (evaluation of likelihood of externally provided data) needs to be fast
    Therefore we invert the Flow

    The parameter sizes for transforming a distribution of dimension d:
    shape u = (1, d)
    shape w = (1, d)
    shape b = (1, 1)
    """
    _u, _w, _b = None, None, None

    def __init__(self, params, n_dims, validate_args=False, name='Inverted_Planar_Flow'):
        """
        Initializes a planar flow for transforming a distribution of dimension n_dims
        :param params: Tensor shape (?, 2*n_dims+1). This will be split into the individual parameters u (?, n_dims), w (?, n_dims), b (?, 1)
        The parameters will be constrained to make the flow invertible, as suggested by Rezende, Mohamed 2015
        :param n_dims: The dimension of the distribution that will be transformed
        """

        super(InvertedPlanarFlow, self).__init__(params, n_dims, validate_args=validate_args, name=name)

        # split the input parameter in to the individual parameters u, w, b
        u_index, w_index, b_index = 0, 1, 2
        flow_params = [InvertedPlanarFlow._handle_input_dimensionality(x)
                       for x in tf.split(value=params, num_or_size_splits=[n_dims, n_dims, 1], axis=1)]

        # constrain u before assigning it
        self._u = InvertedPlanarFlow._u_circ(flow_params[u_index], flow_params[w_index])
        self._w = flow_params[w_index]
        self._b = flow_params[b_index]

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow
        :return: (int) The dimension of the parameter space for this flow, n_dims + n_dims + 1
        """
        return n_dims + n_dims + 1

    @staticmethod
    def _u_circ(u, w):
        """
        To ensure invertibility of the flow, the following condition needs to hold: w_t * u >= -1
        :return: The transformed u
        """
        wtu = tf.reduce_sum(w*u, 1, keepdims=True)
        m_wtu = -1. + tf.nn.softplus(wtu) + 1e-4
        norm_w_squared = tf.reduce_sum(w**2, 1, keepdims=True)
        return u + (m_wtu - wtu)*(w/norm_w_squared)

    def _wzb(self, z):
        """
        Computes w_t * z + b
        """
        return tf.reduce_sum(self._w * z, 1, keepdims=True) + self._b

    @staticmethod
    def _der_tanh(z):
        """
        Computes the derivative of hyperbolic tangent
        """
        return 1. - tf.tanh(z) ** 2

    def _inverse(self, z):
        """
        Runs a backward pass through the bijector
        Also checks for whether the flow is actually invertible
        """
        z = InvertedPlanarFlow._handle_input_dimensionality(z)
        invertible = tf.assert_greater_equal(tf.reduce_sum(self._w * self._u, 1), -1., name='Invertibility_Constraint')
        with tf.control_dependencies([invertible]):
            return z + self._u * tf.tanh(self._wzb(z))

    def _forward(self, x):
        """ As our flows are inverted, sampling / inverting is slow / impossible"""
        raise NotImplementedError()

    def _ildj(self, z):
        """
        Computes the ln of the absolute determinant of the jacobian
        """
        z = InvertedPlanarFlow._handle_input_dimensionality(z)
        psi = self._der_tanh(self._wzb(z)) * self._w
        det_grad = 1. + tf.reduce_sum(self._u * psi, 1, keepdims=True)
        return tf.log(tf.abs(det_grad))


