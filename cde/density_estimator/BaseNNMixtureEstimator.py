import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
import tensorflow as tf
import sklearn

from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable
import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.map_inference import MAP_inference


class BaseNNMixtureEstimator(LayersPowered, Serializable, BaseDensityEstimator):

  def mean_(self, x_cond, n_samples=10**7):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert hasattr(self, '_get_mixture_components')
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    means = np.zeros((x_cond.shape[0], self.ndim_y))
    weights, locs, _ = self._get_mixture_components(x_cond)
    assert weights.ndim == 2 and locs.ndim == 3
    for i in range(x_cond.shape[0]):
      # mean of density mixture is weights * means of density components
      means[i, :] = weights[i].dot(locs[i])
    return means

  def covariance(self, x_cond, n_samples=None):
    """ Covariance of the fitted distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))

    # compute global mean_of mixture model
    glob_mean = self.mean_(x_cond)

    weights, locs, scales = self._get_mixture_components(x_cond)

    for i in range(x_cond.shape[0]):
      c1 = np.diag(weights[i].dot(scales[i]**2))

      c2 = np.zeros(c1.shape)
      for j in range(weights.shape[1]):
        a = (locs[i][j] - glob_mean[i])
        d = weights[i][j] * np.outer(a,a)
        c2 += d
      covs[i] = c1 + c2

    return covs

  def sample(self, X):
    """ sample from the conditional mixture distributions - requires the model to be fitted

      Args:
        X: values to be conditioned on when sampling - numpy array of shape (n_instances, n_dim_x)

      Returns: tuple (X, Y)
        - X - the values to conditioned on that were provided as argument - numpy array of shape (n_samples, ndim_x)
        - Y - conditional samples from the model p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    assert self.can_sample

    X = self._handle_input_dimensionality(X)

    if np.all(np.all(X == X[0, :], axis=1)):
      return self._sample_rows_same(X)
    else:
      return self._sample_rows_individually(X)

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
         resolution: integer specifying the resolution of evaluation_runs grid

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

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of a GMM. Only if ndim_y = 1

        Based on formulas from section 2.3.2 in "Expected shortfall for distributions in finance",
        Simon A. Broda, Marc S. Paolella, 2011

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
         alpha: quantile percentage of the distribution

       Returns:
         CVaR values for each x to condition on - numpy array of shape (n_values)
       """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
    return self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)

  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 7):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo evaluation

        Returns:
          - VaR values for each x to condition on - numpy array of shape (n_values)
          - CVaR values for each x to condition on - numpy array of shape (n_values)
        """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
    CVaRs = self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)

    assert VaRs.shape == CVaRs.shape == (len(x_cond),)
    return VaRs, CVaRs

  def _conditional_value_at_risk_mixture(self, VaRs, x_cond, alpha=0.01,):
    """
    Based on formulas from section 2.3.2 in "Expected shortfall for distributions in finance",
    Simon A. Broda, Marc S. Paolella, 2011
    """

    weights, locs, scales = self._get_mixture_components(x_cond)

    locs = locs.reshape(locs.shape[:2])
    scales = scales.reshape(scales.shape[:2])

    CVaRs = np.zeros(x_cond.shape[0])

    c = (VaRs[:, None] - locs) / scales
    for i in range(x_cond.shape[0]):
      cdf = norm.cdf(c[i])
      pdf = norm.pdf(c[i])

      # mask very small values to avoid numerical instabilities
      cdf = np.ma.masked_where(cdf < 10 ** -64, cdf)
      pdf = np.ma.masked_where(pdf < 10 ** -64, pdf)

      CVaRs[i] = np.sum((weights[i] * cdf / alpha) * (locs[i] - scales[i] * (pdf / cdf)))
    return CVaRs

  def _sample_rows_same(self, X):
    """ uses efficient sklearn implementation to sample from gaussian mixture -> only works if all rows of X are the same"""
    weights, locs, scales = self._get_mixture_components(np.expand_dims(X[0], axis=0))

    # make sure that sum of weights < 1
    weights = weights.astype(np.float64)
    weights = weights / np.sum(weights)

    gmm = GaussianMixture(n_components=self.n_centers, covariance_type='diag', max_iter=5, tol=1e-1)
    gmm.fit(np.random.normal(size=(100,self.ndim_y))) # just pretending a fit
    # overriding the GMM parameters with own params
    gmm.converged_ = True
    gmm.weights_ = weights[0]
    gmm.means_ = locs[0]
    gmm.covariances_ = scales[0]
    y_sample, _ = gmm.sample(X.shape[0])
    assert y_sample.shape == (X.shape[0], self.ndim_y)
    return X, y_sample

  def _sample_rows_individually(self, X):
    weights, locs, scales = self._get_mixture_components(X)

    Y = np.zeros(shape=(X.shape[0], self.ndim_y))
    for i in range(X.shape[0]):
      idx = np.random.choice(range(self.n_centers), p=weights[i, :])
      Y[i, :] = np.random.normal(loc=locs[i, idx, :], scale=scales[i, idx, :])
    assert X.shape[0] == Y.shape[0]
    return X, Y

  def cdf(self, X, Y):
    """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
         conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )

    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    assert hasattr(self, '_get_mixture_components'), "cdf computation requires _get_mixture_components method"

    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)

    weights, locs, scales = self._get_mixture_components(X)

    P = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      for j in range(self.n_centers):
        P[i] += weights[i, j] * multivariate_normal.cdf(Y[i], mean=locs[i,j,:], cov=np.diag(scales[i,j,:]))
    return P

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
      original_params = self.get_configuration()

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

          model = self.__class__(self.name, self.ndim_x, self.ndim_y)
          model.set_params(**original_params).set_params(**p)

          model.fit(X_train, Y_train, verbose=False)
          scores.append(model.score(X_test, Y_test))

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

  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    assert (self.ndim_x == 1 and X.ndim == 1) or (X.ndim == 2 and X.shape[1] == self.ndim_x), "expected X to have shape (?, %i) but received %s"%(self.ndim_x, str(X.shape))
    assert (Y is None) or (self.ndim_y == 1 and Y.ndim == 1) or (Y.ndim == 2 and Y.shape[1] == self.ndim_y), "expected Y to have shape (?, %i) but received %s"%(self.ndim_y, str(Y.shape))
    return BaseDensityEstimator._handle_input_dimensionality(self, X, Y, fitting=fitting)

  def _compute_data_normalization(self, X, Y):
    # compute data statistics (mean & std)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)

    self.data_statistics = {
      'X_mean': X_mean,
      'X_std': X_std,
      'Y_mean': Y_mean,
      'Y_std': Y_std,
    }

    # assign them to tf variables
    sess = tf.get_default_session()
    sess.run([
      tf.assign(self.mean_x_sym, X_mean),
      tf.assign(self.std_x_sym, X_std),
      tf.assign(self.mean_y_sym, Y_mean),
      tf.assign(self.std_y_sym, Y_std)
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

  def _setup_inference_and_initialize(self):
    # setup inference procedure
    with tf.variable_scope(self.name):
      # setup inference procedure
      self.inference = MAP_inference(scope=self.name, data={self.mixture: self.y_input})
      optimizer = tf.train.AdamOptimizer(5e-3)
      self.inference.initialize(var_list=tf.trainable_variables(scope=self.name), optimizer=optimizer, n_iter=self.n_training_epochs)

    self.sess = tf.get_default_session()

    # initialize variables in scope
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    tf.initializers.variables(var_list, name='init').run()

  def __getstate__(self):
    state = LayersPowered.__getstate__(self)
    state['fitted'] = self.fitted
    return state

  def __setstate__(self, state):
    LayersPowered.__setstate__(self, state)
    self.fitted = state['fitted']
    self.sess = tf.get_default_session()

  def _check_uniqueness_of_scope(self, name):
    current_scope = tf.get_variable_scope().name
    scopes = set([variable.name.split('/')[0] for variable in tf.global_variables(scope=current_scope)])
    assert name not in scopes, "%s is already in use for a tensorflow scope - please choose another estimator name"%name