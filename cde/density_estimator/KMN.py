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


from cde.helpers import sample_center_points
from .BaseDensityEstimator import BaseMixtureEstimator

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)


class KernelMixtureNetwork(LayersPowered, Serializable, BaseMixtureEstimator):
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
          random_seed: (optional) seed (int) of the random number generators used
      """

  def __init__(self, name, ndim_x, ndim_y, center_sampling_method='k_means', n_centers=200, keep_edges=False,
               init_scales='default', hidden_sizes=(32, 32), hidden_nonlinearity=tf.nn.tanh, train_scales=True,
               n_training_epochs=1000, x_noise_std=None, y_noise_std=None, random_seed=None):

    Serializable.quick_init(self, locals())

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
    self.center_sampling_method = center_sampling_method
    self.keep_edges = keep_edges

    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std

    if init_scales == 'default':
        init_scales = np.array([1])

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
    LayersPowered.__init__(self, [self.core_output_layer, self.locs_layer, self.scales_layer])

  def fit(self, X, Y, random_seed=None, verbose=True, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        n_folds: number of cross-validation folds (positive integer)
        verbose: (boolean) controls the verbosity (console output)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    # sample locations
    sampled_locs = sample_center_points(Y, method=self.center_sampling_method, k=self.n_centers,
                                     keep_edges=self.keep_edges)



    # setup inference procedure
    self.inference = ed.MAP(data={self.mixtures: self.y_input})
    self.inference.initialize(var_list=tf.trainable_variables(), n_iter=self.n_training_epochs)
    tf.global_variables_initializer().run()
    self.sess = tf.get_default_session()
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

  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """
    assert self.fitted, "model must be fitted to compute likelihood score"

    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
    likelihoods, X_fed, y_fed = self.sess.run([self.likelihoods, self.X_ph, self.Y_ph], feed_dict={self.X_ph: X, self.Y_ph: Y})

    return likelihoods

  def predict_density(self, X, Y=None, resolution=100):
    """ Computes conditional density p(y|x) over a predefined grid of y target values

      Args:
         X: values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
         Y: (optional) y values to be evaluated from p(y|x) -  if not set, Y will be a grid with with specified resolution
         resulution: integer specifying the resolution of evaluation_runs grid

       Returns: tuple (P, Y)
          - P - density p(y|x) - shape (n_instances, resolution**n_dim_y)
          - Y - grid with with specified resolution - shape (resolution**n_dim_y, n_dim_y) or a copy of Y \
            in case it was provided as argument
    """
    assert self.K.learning_phase() == 0
    if Y is None:
        max_scale = np.max(self.sess.run(self.scales))
        Y = np.linspace(self.y_min - 2.5 * max_scale, self.y_max + 2.5 * max_scale, num=resolution)
    X = self._handle_input_dimensionality(X)
    densities, X_fed, y_fed = self.sess.run(self.densities, self.X_ph, self.Y_ph, feed_dict={self.X_ph: X, self.y_grid_ph: Y})

    return densities

  def _build_model(self):
    """
    implementation of the KMN
    """
    with tf.variable_scope(self.name):
      # Input_Layers & placeholders
      self.X_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_x))
      self.Y_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_y))
      self.train_phase = tf.placeholder_with_default(tf.Variable(False), None)

      layer_in_x = L.InputLayer(shape=(None, self.ndim_x), input_var=self.X_ph, name="input_x")
      layer_in_y = L.InputLayer(shape=(None, self.ndim_y), input_var=self.Y_ph, name="input_y")

      # add noise layer if desired
      if self.x_noise_std is not None:
        layer_in_x = L.GaussianNoiseLayer(layer_in_x, self.x_noise_std, noise_on_ph=self.train_phase)
      if self.y_noise_std is not None:
        layer_in_y = L.GaussianNoiseLayer(layer_in_y, self.y_noise_std, noise_on_ph=self.train_phase)

      # get batch size
      self.batch_size = tf.shape(self.X_ph)[0]

      # create core multi-layer perceptron
      core_network = MLP(
        name="core_network",
        input_layer=layer_in_x,
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
      self.y_input = L.get_output(layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc in self.locs_array for scale in scales_array]
      self.mixtures = mixtures = Mixture(cat=cat, components=components, value=tf.zeros_like(self.Y_ph))

      # tensor to store samples
      self.samples = mixtures.sample()

      # placeholder for the grid
      self.y_grid_ph = y_grid_ph = tf.placeholder(tf.float32)
      # tensor to store grid point densities
      self.densities = tf.transpose(mixtures.prob(tf.reshape(y_grid_ph, (-1, 1))))

      # tensor to compute likelihoods
      self.likelihoods = mixtures.prob(self.Y_ph)

  def _param_grid(self):
    n_centers = [int(self.n_samples / 10), 50, 20, 10, 5]

    param_grid = {
        "n_centers": n_centers,
        "center_sampling_method": ["k_means", "random"],
        "keep_edges": [True, False]
    }
    return param_grid

  def fit_by_cv(self, X, Y, n_folds=3, param_grid=None, random_state=None):
    """ Fits the conditional density model with hyperparameter search and cross-validation.

    - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
      the conditional log-likelihood is used for evaluation_runs.
    - Fits the model with the previously selected hyperparameter configuration

    Args:
      X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_samples, n_dim_y)
      n_folds: number of cross-validation folds (positive integer)
      param_grid: (optional) a dictionary with the hyperparameters of the model as key and and a list of respective \
                  parametrizations as value. The hyperparameter search is performed over the cartesian product of \
                  the provided lists.

                  Example:
                  {"n_centers": [20, 50, 100, 200],
                   "center_sampling_method": ["agglomerative", "k_means", "random"],
                   "keep_edges": [True, False]
                  }
      random_state: (int) seed used by the random number generator for shuffeling the data

    """
    original_params = self.get_params()

    if param_grid is None:
      param_grid = self._param_grid()

    param_iterator = sklearn.model_selection.GridSearchCV(self, param_grid, fit_params=None, cv=n_folds)._get_param_iterator()
    cv_scores = []

    for p in param_iterator:
      cv = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

      scores = []
      for train_idx, test_idx in cv.split(X, Y):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        kmn_model = KernelMixtureNetwork()
        kmn_model.set_params(**original_params).set_params(**p)

        kmn_model.fit(X_train, Y_train, verbose=False)
        scores.append(kmn_model.score(X_test, Y_test))

      cv_score = np.mean(scores)
      cv_scores.append(cv_score)

      print("Completed cross-validation of model with params: {}".format(p))
      print("Avg. conditional log likelihood: {}".format(cv_score))

    # Determine parameter set with best conditional likelihood
    best_idx = np.argmax(cv_scores)
    selected_params = param_iterator[best_idx]

    print("Completed grid search - Selected params: {}".format(selected_params))
    print("Refitting model with selected params")

    # Refit with best parameter set
    self.set_params(**selected_params)
    self.fit(X,Y, verbose=False)

  def _get_mixture_components(self, X):
    assert self.fitted

    locs, weights, scales = self.sess.run([self.locs, self.weights, self.scales], feed_dict={self.X_ph: X})

    locs = np.tile(locs.reshape([1] + list(locs.shape)), (X.shape[0], scales.shape[0],1))

    scale_diags = np.tile(scales.reshape((scales.shape[0],1)), (1, self.ndim_y))
    cov = np.concatenate([scale_diags[i,:].reshape((1,1,self.ndim_y)) for i in range(scales.shape[0]) for j in range(self.n_centers)], axis=1)
    cov = np.tile(cov, (X.shape[0], 1, 1))

    return weights, locs, cov

  def __str__(self):
    return "\nEstimator type: {}\n center sampling method: {}\n n_centers: {}\n keep_edges: {}\n init_scales: {}\n train_scales: {}\n " \
             "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n".format(self.__class__.__name__, self.center_sampling_method, self.n_centers,
                                                  self.keep_edges, self.init_scales, self.train_scales, self.n_training_epochs, self.x_noise_std,
                                                                                  self.y_noise_std)


  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    assert (self.ndim_x == 1 and X.ndim == 1) or (X.ndim == 2 and X.shape[1] == self.ndim_x), "expected X to have shape (?, %i) but received %s"%(self.ndim_x, str(X.shape))
    assert (Y is None) or (self.ndim_y == 1 and X.ndim == 1) or (Y.ndim == 2 and Y.shape[1] == self.ndim_y), "expected Y to have shape (?, %i) but received %s"%(self.ndim_y, str(Y.shape))
    return BaseMixtureEstimator._handle_input_dimensionality(self, X, Y, fitting=fitting)

  def __getstate__(self):
    state = LayersPowered.__getstate__(self)
    state['fitted'] = self.fitted
    return state

  def __setstate__(self, state):
    LayersPowered.__setstate__(self, state)
    self.fitted = state['fitted']
    self.sess = tf.get_default_session()
