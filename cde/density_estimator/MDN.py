import numpy as np
import sklearn
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
from scipy.stats import multivariate_normal
from keras.layers import Dense, Dropout, Reshape
#import matplotlib.pyplot as plt

from .helpers import sample_center_points
from .base import BaseMixtureEstimator



class MixtureDensityNetwork(BaseMixtureEstimator):
  """ Mixture Density Network Estimator

    See "Machine Learning and Pattern Recognition", Bishop 2008

    Args:
        n_centers: Number of Gaussian mixture components
        estimator: Keras or tensorflow network that ends with a dense layer to place kernel mixture output on top off,
                   if None use a standard 15 -> 15 Dense network
        X_ph: Placeholder for input to your custom estimator, currently only supporting one input placeholder,
              but should be easy to extend to a list of placeholders
        n_training_epochs: Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
    """


  def __init__(self, n_centers=20, estimator=None, X_ph=None, n_training_epochs=1000, x_noise_std=None, y_noise_std=None):

    self.n_centers = n_centers

    self.estimator = estimator
    self.X_ph = X_ph

    self.train_loss = np.empty(0)
    self.test_loss = np.empty(0)

    self.n_training_epochs = n_training_epochs

    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std

    self.fitted = False
    self.can_sample = True

  def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        n_folds: number of cross-validation folds (positive integer)
        verbose: (boolean) controls the verbosity (console output)

    """

    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    # define the full model
    self._build_model(X, Y)

    # setup inference procedure
    self.inference = ed.MAP(data={self.mixture: self.y_input})
    optimizer = tf.train.AdamOptimizer(5e-3)
    self.inference.initialize(var_list=tf.trainable_variables(), optimizer=optimizer, n_iter=self.n_training_epochs)

    self.sess = ed.get_session()
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
         resulution: integer specifying the resolution of evaluation grid

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

  def sample(self, X):
    """ sample from the conditional mixture distributions - requires the model to be fitted

    Args:
      X: values to be conditioned on when sampling - numpy array of shape (n_instances, n_dim_x)

    Returns: tuple (X, Y)
      - X - the values to conditioned on that were provided as argument - numpy array of shape (n_samples, ndim_x)
      - Y - conditional samples from the model p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    X = self._handle_input_dimensionality(X)
    return X, self.sess.run(self.samples, feed_dict={self.X_ph: X})

  def fit_by_cv(self, X, Y, n_folds=3, param_grid=None, random_state=None):
      """ Fits the conditional density model with hyperparameter search and cross-validation.

      - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
        the conditional log-likelihood is used for evaluation.
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

  def _build_model(self, X, Y):
    """
    implementation of the MDN
    """
    # create a placeholders
    self.Y_ph = tf.placeholder(tf.float32, [None, self.ndim_y])
    self.n_sample_ph = tf.placeholder(tf.int32, None)
    self.train_phase = tf.placeholder_with_default(tf.Variable(False), None)

    # Gaussian noise over Y during training
    if self.y_noise_std:
      y_noised = self.Y_ph + tf.random_normal(tf.shape(self.Y_ph), stddev=self.y_noise_std)
      self.y_input = tf.cond(self.train_phase, lambda: y_noised, lambda: self.Y_ph)
    else:
      self.y_input = self.Y_ph

    # if no external estimator is provided, create a default neural network
    if self.estimator is None:
      self.X_ph = tf.placeholder(tf.float32, [None, self.ndim_x])
      # Gaussian noise over input X during training
      if self.x_noise_std:
        x_noised = self.X_ph + tf.random_normal(tf.shape(self.X_ph), stddev=self.x_noise_std)
        x_input = tf.cond(self.train_phase, lambda: x_noised, lambda: self.X_ph)
      else:
        x_input = self.X_ph
      # two dense hidden layers with 15 nodes each
      net = Dense(15, activation='elu')(x_input)
      net = Dense(15, activation='elu')(net)
      self.estimator = net

    # locations and scales of the mixture components
    self.locs = Dense(self.n_centers * self.ndim_y)(net)
    #self.locs = tf.layers.dense(net, self.n_centers * self.ndim_y, activation=None)
    self.locs = locs = tf.reshape(self.locs, (-1, self.n_centers, self.ndim_y))

    self.scales = Dense(self.n_centers * self.ndim_y, activation='softplus')(net)
    #self.scales = tf.layers.dense(net, self.n_centers * self.ndim_y, activation=tf.exp)
    self.scales = scales = tf.reshape(self.scales, (-1, self.n_centers, self.ndim_y))

    # put mixture components together

    self.logits = logits = Dense(self.n_centers)(net)
    #self.logits = logits = tf.layers.dense(net, self.n_centers, activation=None)
    self.weights = tf.nn.softmax(logits)
    self.cat = cat = Categorical(logits=logits)
    self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                  in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]
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




