from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
import numpy as np
import tensorflow as tf

from cde.density_estimator.BaseNNEstimator import BaseNNEstimator
from cde.utils.tf_utils.map_inference import MAP_inference
from cde.utils.tf_utils.adamW import AdamWOptimizer


class BaseNNMixtureEstimator(BaseNNEstimator):
  weight_decay = 0.0

  def mean_(self, x_cond, n_samples=None):
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

  def std_(self, x_cond, n_samples=10 ** 6):
    """ Standard deviation of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Standard deviations  sqrt(Var[y|x]) corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    covs = self.covariance(x_cond, n_samples=n_samples)
    return np.sqrt(np.diagonal(covs, axis1=1, axis2=2))

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

  def mean_std(self, x_cond, n_samples=None):
    """ Computes Mean and Covariance of the fitted distribution conditioned on x_cond.
        Computationally more efficient than calling mean and covariance computatio separately

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] and Covariances Cov[y|x]
    """
    mean = self.mean_(x_cond, n_samples=n_samples)
    std = self.std_(x_cond, n_samples=n_samples)
    return mean, std

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
          n_samples: number of samples for monte carlo model_fitting

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

  def _partial_fit(self, X, Y, n_epoch=1, eval_set=None, verbose=True):
    """
    update model
    """
    # loop over epochs
    for i in range(n_epoch):

      # run inference, update trainable variables of the model
      info_dict = self.inference.update(feed_dict={self.X_ph: X, self.Y_ph: Y, self.train_phase: True,
                                                   self.dropout_ph: self.dropout})

      # compute evaluation loss
      if eval_set is not None:
        X_test, Y_test = eval_set
        info_dict['eval_loss'] = self.sess.run(self.inference.loss, feed_dict={self.X_ph: X_test, self.Y_ph: Y_test})

      # only print progress for the initial fit, not for additional updates
      if not self.fitted and verbose:
        self.inference.progbar.update(info_dict.pop('t'), info_dict)

    if verbose:
      train_loss = info_dict['loss'] / len(Y)
      print("mean log-loss train: {:.4f}".format(train_loss))
      if eval_set is not None:
        test_loss = info_dict['eval_loss'] / len(Y_test)
        print("mean log-loss valid: {:.4f}".format(test_loss))

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

  def _add_softmax_entropy_regularization(self):
      # softmax entropy penalty -> regularization
      self.softmax_entropy = tf.reduce_mean(tf.reduce_sum(- tf.multiply(tf.log(self.weights), self.weights), axis=1))
      self.entropy_reg_coef_ph = tf.placeholder_with_default(float(self.entropy_reg_coef), name='entropy_reg_coef',
                                                             shape=())
      self.softmax_entrop_loss = self.entropy_reg_coef_ph * self.softmax_entropy
      tf.losses.add_loss(self.softmax_entrop_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

  def _sample_rows_individually(self, X):
    weights, locs, scales = self._get_mixture_components(X)

    assert locs.shape[1] == scales.shape[1] == weights.shape[1]

    Y = np.zeros(shape=(X.shape[0], self.ndim_y))
    for i in range(X.shape[0]):
      idx = np.random.choice(range(locs.shape[1]), p=weights[i, :])
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

  def reset_fit(self):
    """
    resets all tensorflow objects and
    :return:
    """
    tf.reset_default_graph()
    self._build_model()
    self.fitted = False

  def _setup_inference_and_initialize(self):
    # setup inference procedure
    with tf.variable_scope(self.name):
      # setup inference procedure
      self.inference = MAP_inference(scope=self.name, data={self.mixture: self.y_input})
      optimizer = AdamWOptimizer(weight_decay=self.weight_decay, learning_rate=5e-3) if self.weight_decay \
        else tf.train.AdamOptimizer(learning_rate=2e-3)
      self.inference.initialize(var_list=tf.trainable_variables(scope=self.name), optimizer=optimizer, n_iter=self.n_training_epochs)

    self.sess = tf.get_default_session()

    # initialize variables in scope
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    tf.initializers.variables(var_list, name='init').run()
