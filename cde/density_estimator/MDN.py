import numpy as np
import sklearn
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Mixture, MultivariateNormalDiag
from cde.utils.tf_utils.network import MLP
import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable
from cde.utils.tf_utils.map_inference import MAP_inference

#import matplotlib.pyplot as plt

from .BaseNNMixtureEstimator import BaseNNMixtureEstimator

class MixtureDensityNetwork(BaseNNMixtureEstimator):
  """ Mixture Density Network Estimator

    See "Mixture Density networks", Bishop 1994

    Args:
        name: (str) name space of MDN (should be unique in code, otherwise tensorflow namespace collitions may arise)
        ndim_x: (int) dimensionality of x variable
        ndim_y: (int) dimensionality of y variable
        n_centers: Number of Gaussian mixture components
        hidden_sizes: (tuple of int) sizes of the hidden layers of the neural network
        hidden_nonlinearity: (tf function) nonlinearity of the hidden layers
        n_training_epochs: Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
        entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
        weight_normalization: (boolean) whether weight normalization shall be used
        data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
        random_seed: (optional) seed (int) of the random number generators used
    """


  def __init__(self, name, ndim_x, ndim_y, n_centers=20, hidden_sizes=(16, 16), hidden_nonlinearity=tf.nn.tanh, n_training_epochs=1000,
               x_noise_std=None, y_noise_std=None, entropy_reg_coef=0.0, weight_normalization=True, data_normalization=True, random_seed=None):

    Serializable.quick_init(self, locals())
    self._check_uniqueness_of_scope(name)

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y

    self.random_state = np.random.RandomState(seed=random_seed)
    tf.set_random_seed(random_seed)

    self.n_centers = n_centers

    self.hidden_sizes = hidden_sizes
    self.hidden_nonlinearity = hidden_nonlinearity

    self.train_loss = np.empty(0)
    self.test_loss = np.empty(0)

    self.n_training_epochs = n_training_epochs

    # regularization parameters
    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std
    self.entropy_reg_coef = entropy_reg_coef
    self.weight_normalization = weight_normalization
    self.data_normalization = data_normalization

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.softmax_layer_weights, self.softplus_layer_scales, self.reshape_layer_locs, self.layer_in_y])

  def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        n_folds: number of cross-validation folds (positive integer)
        verbose: (boolean) controls the verbosity (console output)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    self._setup_inference_and_initialize()

    # data normalization if desired
    if self.data_normalization: # this must happen after the initialization
      self._compute_data_normalization(X, Y)  # computes mean & std of data and assigns it to tf graph for normalization

    # train the model
    self._partial_fit(X, Y, n_epoch=self.n_training_epochs, verbose=verbose, **kwargs)
    self.fitted = True

  def _partial_fit(self, X, Y, n_epoch=1, eval_set=None, verbose=True):
    """
    update model
    """

    # loop over epochs
    for i in range(n_epoch):

        # run inference, update trainable variables of the model
        info_dict = self.inference.update(feed_dict={self.X_ph: X, self.Y_ph: Y, self.train_phase: True})

        train_loss = info_dict['loss'] / len(Y)
        self.train_loss = np.append(self.train_loss, -train_loss)

        if eval_set is not None:
            X_test, y_test = eval_set
            test_loss = self.sess.run(self.inference.loss, feed_dict={self.X_ph: X_test, self.y_ph: y_test}) / len(y_test)
            self.test_loss = np.append(self.test_loss, -test_loss)

        #scales = self.sess.run(self.scales, feed_dict={self.X_ph: X[:2,:]})
        #print(scales[1])

        # only print progress for the initial fit, not for additional updates
        if not self.fitted and verbose:
            self.inference.print_progress(info_dict)

    if verbose:
      print("mean log-loss train: {:.3f}".format(train_loss))
      if eval_set is not None:
          print("mean log-loss test: {:.3f}".format(test_loss))

  def _build_model(self):
    """
    implementation of the MDN
    """

    with tf.variable_scope(self.name):
      self.layer_in_x, self.layer_in_y = self._build_input_layers()  # add playeholders, data_normalization and data_noise if desired
      # create core multi-layer perceptron
      mlp_output_dim = 2 * self.ndim_y * self.n_centers + self.n_centers
      core_network = MLP(
              name="core_network",
              input_layer=self.layer_in_x,
              output_dim=mlp_output_dim,
              hidden_sizes=self.hidden_sizes,
              hidden_nonlinearity=self.hidden_nonlinearity,
              output_nonlinearity=None,
              weight_normalization=self.weight_normalization
          )

      core_output_layer = core_network.output_layer

      # slice output of MLP into three equally sized parts for loc, scale and mixture weights
      slice_layer_locs = L.SliceLayer(core_output_layer, indices=slice(0, self.ndim_y * self.n_centers), axis=-1)
      slice_layer_scales = L.SliceLayer(core_output_layer, indices=slice(self.ndim_y * self.n_centers, 2 * self.ndim_y * self.n_centers), axis=-1)
      slice_layer_weights = L.SliceLayer(core_output_layer, indices=slice(2 * self.ndim_y * self.n_centers, mlp_output_dim), axis=-1)

      # locations mixture components
      self.reshape_layer_locs = L.ReshapeLayer(slice_layer_locs, (-1, self.n_centers, self.ndim_y))
      self.locs = L.get_output(self.reshape_layer_locs)

      # scales of the mixture components
      reshape_layer_scales = L.ReshapeLayer(slice_layer_scales, (-1, self.n_centers, self.ndim_y))
      self.softplus_layer_scales = L.NonlinearityLayer(reshape_layer_scales, nonlinearity=tf.nn.softplus)
      self.scales = L.get_output(self.softplus_layer_scales)

      # weights of the mixture components
      self.logits = L.get_output(slice_layer_weights)
      self.softmax_layer_weights = L.NonlinearityLayer(slice_layer_weights, nonlinearity=tf.nn.softmax)
      self.weights = L.get_output(self.softmax_layer_weights)

      # # put mixture components together
      self.y_input = L.get_output(self.layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                     in zip(tf.unstack(self.locs, axis=1), tf.unstack( self.scales, axis=1))]
      self.mixture = mixture = Mixture(cat=cat, components=components, value=tf.zeros_like(self.y_input))

      # softmax entropy penalty -> regularization
      self.softmax_entropy = tf.reduce_sum(- tf.multiply(tf.log(self.weights), self.weights), axis=1)
      self.entropy_reg_coef_ph = tf.placeholder_with_default(float(self.entropy_reg_coef), name='entropy_reg_coef', shape=())
      self.softmax_entrop_loss = self.entropy_reg_coef_ph * self.softmax_entropy
      tf.losses.add_loss(self.softmax_entrop_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

      # tensor to store samples
      self.samples = mixture.sample() #TODO either use it or remove it

      # tensor to compute probabilities
      if self.data_normalization:
        self.pdf_ = mixture.prob(self.y_input) / tf.reduce_prod(self.std_y_sym)
      else:
        self.pdf_ = mixture.prob(self.y_input)

      # symbolic tensors for getting the unnormalized mixture components
      if self.data_normalization:
        self.scales_unnormalized = self.scales * self.std_y_sym
        self.locs_unnormalized = self.locs * self.std_y_sym + self.mean_y_sym
      else:
        self.scales_unnormalized = self.scales
        self.locs_unnormalized = self.locs

  def _param_grid(self):
    n_centers = [1, 2, 4, 8, 16, 32]

    param_grid = {
        "n_centers": n_centers,
    }
    return param_grid

  def _get_mixture_components(self, X):
    assert self.fitted
    weights, locs, scales = self.sess.run([self.weights, self.locs_unnormalized, self.scales_unnormalized], feed_dict={self.X_ph: X})
    assert weights.shape[0] == locs.shape[0] == scales.shape[0] == X.shape[0]
    assert weights.shape[1] == locs.shape[1] == scales.shape[1] == self.n_centers
    assert locs.shape[2] == scales.shape[2] == self.ndim_y
    assert locs.ndim == 3 and scales.ndim == 3 and weights.ndim == 2
    return weights, locs, scales
