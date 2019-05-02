import tensorflow as tf


class BaseNormalizingFlow(tf.distributions.bijectors.Bijector):
    def __init__(self, params, n_dims, validate_args=False, name='BaseNormalizingFlow'):
        """
        Initializes the normalizing flows, checking for a valid parameter size
        :param params: The batched parameters, shape (?, get_param_size(n_dims))
        :param n_dims: The dimension of the distribution that is being transformed
        """
        super(BaseNormalizingFlow, self).__init__(validate_args=validate_args, name=name)
        self.n_dims = n_dims
        assert params.shape[1] == self.get_param_size(n_dims), 'Shape is {}, should be {}'.format(params.shape[1], self.get_param_size(n_dims))
        assert len(params.shape) == 2

    @staticmethod
    def get_param_size(n_dims):
        """
        Returns the size of the parameter space for this normalizing flow as an int
        """
        raise NotImplementedError()

    def _ildj(self, y):
        """
        :param y: shape (batch_size, n_dims)
        :return: the inverse log det jacobian, shape (batch_size, 1)
        """
        raise NotImplementedError()

    def _inverse_log_det_jacobian(self, y):
        """
        Adapts the shape of the ildj to the dimension
        For n_dims > 1 we use MultivariateNormalDistribution, which has the output shape (batch_size, ) for it's pdf
        instead of (batch_size, 1) for the UnivariateNormalDistribution
        -> Remove one dimension from the ildj
        """
        if self.n_dims == 1:
            return self._ildj(y)
        else:
            return tf.squeeze(self._ildj(y), axis=1)

    @staticmethod
    def _handle_input_dimensionality(z):
        """
        If rank(z) is 1, increase rank to 2
        We want tensors of shape (?, N_DIMS)
        """
        return tf.cond(tf.equal(tf.rank(z), tf.rank([0.])), lambda: tf.expand_dims(z, 1), lambda: z)
