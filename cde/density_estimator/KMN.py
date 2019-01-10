#
# code skeleton from https://github.com/janvdvegt/KernelMixtureNetwork
# this version additionally supports fit_by_crossval and multidimentional Y
#
import math
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


from cde.helpers import sample_center_points
from cde.density_estimator.BaseNNMixtureEstimator import BaseNNMixtureEstimator

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)


class KernelMixtureNetwork(BaseNNMixtureEstimator): #TODO: KMN doesn not anymore pass its unittests - find out what's the problem

  # noinspection PyPackageRequirements
  """ Kernel Mixture Network Estimator

      https://arxiv.org/abs/1705.07111

      Args:
          name: (str) name space of MDN (should be unique in code, otherwise tensorflow namespace collitions may arise)
          ndim_x: (int) dimensionality of x variable
          ndim_y: (int) dimensionality of y variable
          center_sampling_method: String that describes the method to use for finding kernel centers. Allowed values \
                                  [all, random, distance, k_means, agglomerative]
          n_centers: Number of kernels to use in the output
          keep_edges: Keep the extreme y values as center to keep expressiveness
          init_scales: List or scalar that describes (initial) values of bandwidth parameter
          train_scales: Boolean that describes whether or not to make the scales trainable
          x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise. Adding noise is
          automatically deactivated during
          y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
          entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
          weight_normalization: boolean specifying whether weight normalization shall be used
                  data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
          random_seed: (optional) seed (int) of the random number generators used
  """

  def __init__(self, name, ndim_x, ndim_y, center_sampling_method='k_means', n_centers=200, keep_edges=False,
               init_scales='default', hidden_sizes=(16, 16), hidden_nonlinearity=tf.nn.tanh, train_scales=True,
               n_training_epochs=1000, x_noise_std=None, y_noise_std=None, entropy_reg_coef=0.0, weight_normalization=True, data_normalization=True, random_seed=None):

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

    # center sampling parameters
    self.center_sampling_method = center_sampling_method
    self.keep_edges = keep_edges

    # regularization parameters
    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std
    self.entropy_reg_coef = entropy_reg_coef
    self.weight_normalization = weight_normalization
    self.data_normalization = data_normalization

    if init_scales == 'default':
        init_scales = np.array([1.0])

    self.n_scales = len(init_scales)
    # Transform scales so that the softplus will result in passed init_scales
    self.init_scales = [math.log(math.exp(s) - 1) for s in init_scales]
    self.train_scales = train_scales

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.core_output_layer, self.locs_layer, self.scales_layer, self.layer_in_y])

  def fit(self, X, Y, random_seed=None, verbose=True, **kwargs):
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
    if self.data_normalization:  # this must happen after the initialization
      self._compute_data_normalization(X, Y)  # computes mean & std of data and assigns it to tf graph for normalization
      Y_normalized = (Y - self.data_statistics['Y_mean']) / (self.data_statistics['Y_std'] + 1e-8)
    else:
      Y_normalized = Y

    # sample locations and assign them to tf locs variable
    sampled_locs = sample_center_points(Y_normalized, method=self.center_sampling_method, k=self.n_centers,
                                     keep_edges=self.keep_edges)
    self.sess.run(tf.assign(self.locs, sampled_locs))

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
        test_loss, X_test_fed, y_test_fed = self.sess.run(self.inference.loss, X_test, y_test, feed_dict={self.X_ph: X_test, self.Y_ph: y_test}) / len(
          y_test)
        self.test_loss = np.append(self.test_loss, -test_loss)

      # only print progress for the initial fit, not for additional updates
      if not self.fitted and verbose:
        self.inference.print_progress(info_dict)

    if verbose:
      print("mean log-loss train: {:.3f}".format(train_loss))
      if eval_set is not None:
        print("mean log-loss test: {:.3f}".format(test_loss))

      print("optimal scales: {}".format(self.sess.run(self.scales)))

  def _build_model(self):
    """
    implementation of the KMN
    """
    with tf.variable_scope(self.name):
      self.layer_in_x, self.layer_in_y = self._build_input_layers() # add playeholders, data_normalization and data_noise if desired

      self.X_in = L.get_output(self.layer_in_x)
      self.Y_in = L.get_output(self.layer_in_y)

      # get batch size
      self.batch_size = tf.shape(self.X_ph)[0]

      # create core multi-layer perceptron
      core_network = MLP(
        name="core_network",
        input_layer=self.layer_in_x,
        output_dim=self.n_centers*self.n_scales,
        hidden_sizes=self.hidden_sizes,
        hidden_nonlinearity=self.hidden_nonlinearity,
        output_nonlinearity=None,
      )

      self.core_output_layer = core_network.output_layer

      # weights of the mixture components
      self.logits = L.get_output(self.core_output_layer)
      self.softmax_layer_weights = L.NonlinearityLayer(self.core_output_layer, nonlinearity=tf.nn.softmax)
      self.weights = L.get_output(self.softmax_layer_weights)

      # locations of the kernelfunctions
      self.locs = tf.Variable(np.zeros((self.n_centers, self.ndim_y)), name="locs", trainable=False, dtype=tf.float32) # assign sampled locs when fitting
      self.locs_layer = L.VariableLayer(core_network.input_layer, (self.n_centers, self.ndim_y), variable=self.locs, name="locs", trainable=False)

      self.locs_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, self.n_centers, self.ndim_y)), self.locs), perm=[1, 0, 2]))
      assert len(self.locs_array) == self.n_centers

      # scales of the gaussian kernels
      log_scales_layer = L.VariableLayer(core_network.input_layer, (self.n_scales,),
                                      variable=tf.Variable(self.init_scales, dtype=tf.float32, trainable=self.train_scales),
                                      name="log_scales", trainable=self.train_scales)

      self.scales_layer = L.NonlinearityLayer(log_scales_layer, nonlinearity=tf.nn.softplus)
      self.scales = L.get_output(self.scales_layer)
      self.scales_array = scales_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, self.ndim_y, self.n_scales)), self.scales), perm=[2,0,1]))
      assert len(self.scales_array) == self.n_scales

      # put mixture components together
      self.y_input = L.get_output(self.layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc in self.locs_array for scale in scales_array]
      self.mixture = mixture = Mixture(cat=cat, components=components)

      # softmax entropy penalty -> regularization
      self.softmax_entropy = tf.reduce_sum(- tf.multiply(tf.log(self.weights), self.weights), axis=1)
      self.entropy_reg_coef_ph = tf.placeholder_with_default(float(self.entropy_reg_coef), name='entropy_reg_coef', shape=())
      self.softmax_entrop_loss = self.entropy_reg_coef_ph * self.softmax_entropy
      tf.losses.add_loss(self.softmax_entrop_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

      # tensor to compute probabilities
      if self.data_normalization:
        self.pdf_ = mixture.prob(self.y_input) / tf.reduce_prod(self.std_y_sym)
      else:
        self.pdf_ = mixture.prob(self.y_input)

      # symbolic tensors for getting the unnormalized mixture components
      if self.data_normalization:
        self.scales_unnormalized = tf.transpose(tf.multiply(tf.ones((self.ndim_y, self.n_scales)), self.scales)) * self.std_y_sym # shape = (n_scales, ndim_y)
        self.locs_unnormalized = self.locs * self.std_y_sym + self.mean_y_sym
      else:
        self.scales_unnormalized = tf.transpose(tf.multiply(tf.ones((self.ndim_y, self.n_scales)), self.scales)) # shape = (n_scales, ndim_y)
        self.locs_unnormalized = self.locs

  def _param_grid(self):
    n_centers = [int(self.n_samples / 10), 50, 20, 10, 5]

    param_grid = {
        "n_centers": n_centers,
        "center_sampling_method": ["k_means", "random"],
        "keep_edges": [True, False]
    }
    return param_grid


  def _get_mixture_components(self, X):
    assert self.fitted

    locs, weights, scales = self.sess.run([self.locs_unnormalized, self.weights, self.scales_unnormalized], feed_dict={self.X_ph: X})

    locs = np.concatenate([np.tile(np.expand_dims(locs[i:i+1], axis=1), (X.shape[0], self.n_scales, 1)) for i in range(self.n_centers)], axis=1)
    cov = np.tile(np.expand_dims(scales, axis=0), (X.shape[0], self.n_centers, 1))

    assert weights.shape[0] == locs.shape[0] == cov.shape[0] == X.shape[0]
    assert weights.shape[1] == locs.shape[1] == cov.shape[1] == self.n_centers*self.n_scales
    assert locs.shape[2] == cov.shape[2] == self.ndim_y
    assert locs.ndim == 3 and cov.ndim == 3 and weights.ndim == 2
    return weights, locs, cov

  def __str__(self):
    return "\nEstimator type: {}\n center sampling method: {}\n n_centers: {}\n keep_edges: {}\n init_scales: {}\n train_scales: {}\n " \
             "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n".format(self.__class__.__name__, self.center_sampling_method, self.n_centers,
                                                  self.keep_edges, self.init_scales, self.train_scales, self.n_training_epochs, self.x_noise_std,
                                                                                  self.y_noise_std)
