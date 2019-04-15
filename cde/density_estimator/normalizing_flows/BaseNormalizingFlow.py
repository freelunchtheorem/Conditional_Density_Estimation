import tensorflow as tf


class BaseNormalizingFlow(tf.distributions.bijectors.Bijector):

    def __init__(self, params, n_dims, validate_args=False, name='BaseNormalizingFlow'):
        super(BaseNormalizingFlow, self).__init__(validate_args=validate_args, name=name)
        assert params.shape[1] == self.get_param_size(n_dims)
        assert len(params.shape) == 2

    @staticmethod
    def get_param_size(n_dims):
        """
        Returns the size of the parameter space for this normalizing flow
        """
        raise NotImplementedError()

    @staticmethod
    def _handle_input_dimensionality(z):
        """
        If rank(z) is 1, increase rank to 2
        We want tensors of shape (?, N_DIMS)
        """
        return tf.cond(tf.equal(tf.rank(z), tf.rank([0.])), lambda: tf.expand_dims(z, 1), lambda: z)
