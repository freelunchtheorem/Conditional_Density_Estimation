import numpy as np
import sklearn
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Mixture, MultivariateNormalDiag
from keras.layers import Dense
from cde.utils.tf_utils.network import MLP
import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable

#import matplotlib.pyplot as plt

from .BaseDensityEstimator import BaseMixtureEstimator



class MixtureDensityNetwork(LayersPowered, Serializable, BaseMixtureEstimator):
  """ Mixture Density Network Estimator

    See "Mixture Density networks", Bishop 1994

    Args:
        n_centers: Number of Gaussian mixture components
        estimator: Keras or tensorflow network that ends with a dense layer to place kernel mixture output on top off,
                   if None use a standard 15 -> 15 Dense network
        X_ph: Placeholder for input to your custom estimator, currently only supporting one input placeholder,
              but should be easy to extend to a list of placeholders
        n_training_epochs: Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
        random_seed: (optional) seed (int) of the random number generators used
    """


  def __init__(self, name, ndim_x, ndim_y, n_centers=20, hidden_sizes=(32, 32), hidden_nonlinearity=tf.nn.tanh, n_training_epochs=1000,
               x_noise_std=None, y_noise_std=None, random_seed=None):

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

    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.softmax_layer_weights, self.softplus_layer_scales, self.reshape_layer_locs])

  def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        n_folds: number of cross-validation folds (positive integer)
        verbose: (boolean) controls the verbosity (console output)

    """

    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    # setup inference procedure
    self.inference = ed.MAP(data={self.mixture: self.y_input})
    optimizer = tf.train.AdamOptimizer(5e-3)
    self.inference.initialize(var_list=tf.trainable_variables(), optimizer=optimizer, n_iter=self.n_training_epochs)

    self.sess = tf.get_default_session()
    tf.global_variables_initializer().run()

    self.can_sample = True
    self.has_cdf = True

    # train the model
    self._partial_fit(X, Y, n_epoch=self.n_training_epochs, verbose=verbose, **kwargs)
    self.fitted = True

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

      p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})

      assert p.ndim == 1 and p.shape[0] == X.shape[0]
      return p

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
    if Y is None:
        max_scale = np.max(self.sess.run(self.scales))
        Y = np.linspace(self.y_min - 2.5 * max_scale, self.y_max + 2.5 * max_scale, num=resolution)
    X = self._handle_input_dimensionality(X)
    return self.sess.run(self.densities, feed_dict={self.X_ph: X, self.y_grid_ph: Y})

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

          kmn_model = self.__class__()
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

      # create core multi-layer perceptron
      mlp_output_dim = 2 * self.ndim_y * self.n_centers + self.n_centers
      core_network = MLP(
              name="core_network",
              input_layer=layer_in_x,
              output_dim=mlp_output_dim,
              hidden_sizes=self.hidden_sizes,
              hidden_nonlinearity=self.hidden_nonlinearity,
              output_nonlinearity=None,
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
      self.y_input = L.get_output(layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                     in zip(tf.unstack(self.locs, axis=1), tf.unstack( self.scales, axis=1))]
      self.mixture = mixture = Mixture(cat=cat, components=components, value=tf.zeros_like(self.y_input))

      # tensor to store samples
      self.samples = mixture.sample()

      # placeholder for the grid
      self.y_grid_ph = y_grid_ph = tf.placeholder(tf.float32)
      # tensor to store grid point densities
      self.densities = tf.transpose(mixture.prob(tf.reshape(y_grid_ph, (-1, 1))))

      # tensor to compute probabilities
      self.pdf_ = mixture.prob(self.y_input)

  def _param_grid(self):
    n_centers = [1, 2, 4, 8, 16, 32]

    param_grid = {
        "n_centers": n_centers,
    }
    return param_grid

  def _get_mixture_components(self, X):
    assert self.fitted
    weights, locs, scales = self.sess.run([self.weights, self.locs, self.scales], feed_dict={self.X_ph: X})
    return weights, locs, scales

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
