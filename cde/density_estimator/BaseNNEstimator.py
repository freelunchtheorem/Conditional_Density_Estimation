import numpy as np
import tensorflow as tf
from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable
import cde.utils.tf_utils.layers as L


class BaseNNEstimator(LayersPowered, Serializable, BaseDensityEstimator):
    """
    Base class for a density estimator using a neural network to parametrize the distribution p(y|x)
    To use this class, implement pdf_, cdf_ and log_pdf_ or overwrite the parent methods
    """

    # input data can be normalized before training
    data_normalization = False

    # used for noise regularization of the data
    x_noise_std = False
    y_noise_std = False

    # was the model fitted to the data or not
    fitted = False

    def pdf(self, X, Y):
        """ Predicts the conditional probability p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              conditional probability p(y|x) - numpy array of shape (n_query_samples, )

         """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

    def cdf(self, X, Y):
        """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
             conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )

        """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.cdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

    def log_pdf(self, X, Y):
        """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              onditional log-probability log p(y|x) - numpy array of shape (n_query_samples, )

         """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.log_pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

    def _compute_data_normalization(self, X, Y):
        # compute data statistics (mean & std)
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.y_mean = np.mean(Y, axis=0)
        self.y_std = np.std(Y, axis=0)

        self.data_statistics = {
            'X_mean': self.x_mean,
            'X_std': self.x_std,
            'Y_mean': self.y_mean,
            'Y_std': self.y_std,
        }

        # assign them to tf variables
        sess = tf.get_default_session()
        sess.run([
            tf.assign(self.mean_x_sym, self.x_mean),
            tf.assign(self.std_x_sym, self.x_std),
            tf.assign(self.mean_y_sym, self.y_mean),
            tf.assign(self.std_y_sym, self.y_std)
        ])

    def _build_input_layers(self):
        # Input_Layers & placeholders
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_x))
        self.Y_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_y))
        self.train_phase = tf.placeholder_with_default(False, None)

        layer_in_x = L.InputLayer(shape=(None, self.ndim_x), input_var=self.X_ph, name="input_x")
        layer_in_y = L.InputLayer(shape=(None, self.ndim_y), input_var=self.Y_ph, name="input_y")

        # add data normalization layer if desired
        if self.data_normalization:
            layer_in_x = L.NormalizationLayer(layer_in_x, self.ndim_x, name="data_norm_x")
            self.mean_x_sym, self.std_x_sym = layer_in_x.get_params()
            layer_in_y = L.NormalizationLayer(layer_in_y, self.ndim_y, name="data_norm_y")
            self.mean_y_sym, self.std_y_sym = layer_in_y.get_params()

        # add noise layer if desired
        if self.x_noise_std is not None:
            layer_in_x = L.GaussianNoiseLayer(layer_in_x, self.x_noise_std, noise_on_ph=self.train_phase)
        if self.y_noise_std is not None:
            layer_in_y = L.GaussianNoiseLayer(layer_in_y, self.y_noise_std, noise_on_ph=self.train_phase)

        return layer_in_x, layer_in_y

    def __getstate__(self):
        state = LayersPowered.__getstate__(self)
        state['fitted'] = self.fitted
        return state

    def __setstate__(self, state):
        LayersPowered.__setstate__(self, state)
        self.fitted = state['fitted']
        self.sess = tf.get_default_session()

    def _handle_input_dimensionality(self, X, Y=None, fitting=False):
        assert (self.ndim_x == 1 and X.ndim == 1) or (X.ndim == 2 and X.shape[1] == self.ndim_x), "expected X to have shape (?, %i) but received %s"%(self.ndim_x, str(X.shape))
        assert (Y is None) or (self.ndim_y == 1 and Y.ndim == 1) or (Y.ndim == 2 and Y.shape[1] == self.ndim_y), "expected Y to have shape (?, %i) but received %s"%(self.ndim_y, str(Y.shape))
        return BaseDensityEstimator._handle_input_dimensionality(self, X, Y, fitting=fitting)

    @staticmethod
    def _check_uniqueness_of_scope(name):
        current_scope = tf.get_variable_scope().name
        scopes = set([variable.name.split('/')[0] for variable in tf.global_variables(scope=current_scope)])
        assert name not in scopes, "%s is already in use for a tensorflow scope - please choose another estimator name"%name

