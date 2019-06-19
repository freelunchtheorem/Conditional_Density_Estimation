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
#import matplotlib.pyplot as plt


from cde.utils.center_point_select import sample_center_points
from cde.density_estimator.BaseNNMixtureEstimator import BaseNNMixtureEstimator

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)


class KernelMixtureNetwork(BaseNNMixtureEstimator):

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
          x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X
          y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y
          adaptive_noise_fn: (callable) that takes the number of samples and the data dimensionality as arguments and returns
                           the noise std as float - if used, the x_noise_std and y_noise_std have no effect
          entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
          weight_decay: (float) the amount of decoupled (http://arxiv.org/abs/1711.05101) weight decay to apply
          l2_reg: (float) the amount of l2 penalty on neural network weights
          l1_reg: (float) the amount of l1 penalty on neural network weights
          weight_normalization: boolean specifying whether weight normalization shall be used
          data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
          dropout: (float) the probability of switching off nodes during training
          random_seed: (optional) seed (int) of the random number generators used
  """

  def __init__(self, name, ndim_x, ndim_y, center_sampling_method='k_means', n_centers=50, keep_edges=True,
               init_scales='default', hidden_sizes=(16, 16), hidden_nonlinearity=tf.nn.tanh, train_scales=True,
               n_training_epochs=1000, x_noise_std=None, y_noise_std=None, adaptive_noise_fn=None,  entropy_reg_coef=0.0,
               weight_decay=0.0, weight_normalization=True, data_normalization=True, dropout=0.0, l2_reg=0.0, l1_reg=0.0,
               random_seed=None):

    Serializable.quick_init(self, locals())
    self._check_uniqueness_of_scope(name)

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y

    self.random_seed = random_seed
    self.random_state = np.random.RandomState(seed=random_seed)
    tf.set_random_seed(random_seed)

    self.n_centers = n_centers

    self.hidden_sizes = hidden_sizes
    self.hidden_nonlinearity = hidden_nonlinearity

    self.n_training_epochs = n_training_epochs

    # center sampling parameters
    self.center_sampling_method = center_sampling_method
    self.keep_edges = keep_edges

    # regularization parameters
    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std
    self.adaptive_noise_fn = adaptive_noise_fn
    self.entropy_reg_coef = entropy_reg_coef
    self.weight_decay = weight_decay
    self.l2_reg = l2_reg
    self.l1_reg = l1_reg
    self.weight_normalization = weight_normalization
    self.data_normalization = data_normalization
    self.dropout = dropout

    if type(init_scales) is str and init_scales == 'default':
        init_scales = np.array([0.7, 0.3])

    self.n_scales = len(init_scales)
    self.train_scales = train_scales
    self.init_scales = init_scales
    # Transform scales so that the softplus will result in passed init_scales
    self.init_scales_softplus = [np.log(np.exp(s) - 1) for s in init_scales]

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

  def fit(self, X, Y, eval_set=None, verbose=True):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        eval_set: (tuple) eval/test set - tuple (X_test, Y_test)
        verbose: (boolean) controls the verbosity (console output)
    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    if eval_set is not None:
      eval_set = self._handle_input_dimensionality(*eval_set)

    self._setup_inference_and_initialize()

    # data normalization if desired
    if self.data_normalization:  # this must happen after the initialization
      self._compute_data_normalization(X, Y)  # computes mean & std of data and assigns it to tf graph for normalization
      Y_normalized = (Y - self.data_statistics['Y_mean']) / (self.data_statistics['Y_std'] + 1e-8)
    else:
      Y_normalized = Y

    self._compute_noise_intensity(X, Y)

    # sample locations and assign them to tf locs variable
    sampled_locs = sample_center_points(Y_normalized, method=self.center_sampling_method, k=self.n_centers,
                                     keep_edges=self.keep_edges, random_state=self.random_state)
    self.sess.run(tf.assign(self.locs, sampled_locs))

    # train the model
    self._partial_fit(X, Y, n_epoch=self.n_training_epochs, eval_set=eval_set, verbose=verbose)
    self.fitted = True

    if verbose:
      print("optimal scales: {}".format(self.sess.run(self.scales)))

  def _build_model(self):
    """
    implementation of the KMN
    """
    with tf.variable_scope(self.name):
      # add placeholders, data_normalization and data_noise if desired. Also sets up the placeholder for dropout prob
      self.layer_in_x, self.layer_in_y = self._build_input_layers()

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
        dropout_ph=self.dropout_ph if self.dropout else None
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
                                         variable=tf.Variable(self.init_scales_softplus, dtype=tf.float32, trainable=self.train_scales),
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

      # regularization
      self._add_softmax_entropy_regularization()
      self._add_l1_l2_regularization(core_network)

      # tensor to compute probabilities
      if self.data_normalization:
        self.pdf_ = mixture.prob(self.y_input) / tf.reduce_prod(self.std_y_sym)
        self.log_pdf_ = mixture.log_prob(self.y_input) - tf.reduce_sum(tf.log(self.std_y_sym))
      else:
        self.pdf_ = mixture.prob(self.y_input)
        self.log_pdf_ = mixture.log_prob(self.y_input)

      # symbolic tensors for getting the unnormalized mixture components
      if self.data_normalization:
        self.scales_unnormalized = tf.transpose(tf.multiply(tf.ones((self.ndim_y, self.n_scales)), self.scales)) * self.std_y_sym # shape = (n_scales, ndim_y)
        self.locs_unnormalized = self.locs * self.std_y_sym + self.mean_y_sym
      else:
        self.scales_unnormalized = tf.transpose(tf.multiply(tf.ones((self.ndim_y, self.n_scales)), self.scales)) # shape = (n_scales, ndim_y)
        self.locs_unnormalized = self.locs

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.core_output_layer, self.locs_layer, self.scales_layer, self.layer_in_y])

  def _param_grid(self):
    param_grid = {
        "n_training_epochs": [500, 1000],
        "n_centers": [50, 200],
        "x_noise_std": [0.15, 0.2, 0.3],
        "y_noise_std": [0.1, 0.15, 0.2]
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
                                                                                  self.keep_edges, self.init_scales_softplus, self.train_scales, self.n_training_epochs, self.x_noise_std,
                                                                                  self.y_noise_std)
