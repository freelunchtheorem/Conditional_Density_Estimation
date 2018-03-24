import time
import numpy as np
import scipy
import logging
from scipy.stats import shapiro, kstest
from cde.density_estimator.base import BaseDensityEstimator
from cde.density_simulation import ConditionalDensity
from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from scipy import stats



class GoodnessOfFit:
  """ Class that takes an estimator a probabilistic simulation model. The estimator is fitted on n_obervation samples.
  Then the goodness of fit w.r.t to the true probability distribution is evaluated

  Args:
    estimator: Estimator instance that implements the functionality from BaseDensityEstimator (can be either fitted or not fitted)
    probabilistic_model: ConditionalDensity instance which implements the methods simulate, pdf, cdf
    n_observations: number of observations which are drawn from the probabilistic_model to fit the estimator.
                    If the estimator is already fitted, this parameter is ignored
    n_x_cond: number of different x to condition on - e.g. if n_x_cond=20, p(y|x) is evaluated for 20 different x
    print_fit_result: boolean that specifies whether the fitted distribution shall be plotted (only works if ndim_x and ndim_y = 1)
    seed: random seed to draw samples from the probabilistic model

  """
  def __init__(self, estimator, probabilistic_model, n_observations=10**5, n_x_cond=10**5, print_fit_result=False, seed=24):

    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, ConditionalDensity), "probabilistic model must inherit from ConditionalDensity"

    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations
    self.n_x_cond = n_x_cond

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    self.seed = seed
    np.random.seed(seed)
    self.X, self.Y = probabilistic_model.simulate(self.n_observations)
    self.X, self.Y = probabilistic_model._handle_input_dimensionality(self.X, self.Y)

    self.time_to_fit = None
    if not estimator.fitted: # fit estimator if necessary
      t_start = time.time()
      estimator.fit(self.X, self.Y)
      self.time_to_fit = (time.time() - t_start) * n_observations / 1000 #time to fit per 1000 samples

    if print_fit_result:
      self.probabilistic_model.plot(mode="pdf")
      estimator.plot()

    self.estimator = estimator


  def kolmogorov_smirnov_cdf(self, x_cond, n_samples=10**7):
    """ Calculates Kolmogorov-Smirnov Statistics

    Args:
      x_cond: x value to condition on
      n_samples: number of samples y drawn from p(y|x_cond)

    Returns:
      (ks_stat, ks_pval) Kolmogorov-Smirnov statistic and corresponding p-pvalue
    """
    X_cond = np.vstack([x_cond for _ in range(n_samples)])
    np.random.seed(self.seed**2)
    _, estimator_conditional_samples = self.estimator.sample(X_cond)
    assert estimator_conditional_samples.ndim == 1 or estimator_conditional_samples.shape[1] == 1 , "Can only compute Kosmogorov Smirnov Statistic of ndim_y = 1"
    estimator_conditional_samples = estimator_conditional_samples.flatten()
    return kstest(estimator_conditional_samples, lambda y: self.probabilistic_model.cdf(X_cond, y))


  def kl_divergence(self, y_res=100, measure_time = False):
    """ Calculates a discrete approximiation of KL_divergence of the fitted ditribution w.r.t. the true distribution

    Args:
      y_res: sampling rate for y
      measure_time: boolean that indicates whether the time_to_predict 1000 density function values shall be returned as second element

    Returns:
      Kl-divergence (float), time_to_predict (float) if measure_time is true
    """
    np.random.seed(self.seed)

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    # prepare mesh
    grid_x = get_variable_grid(self.X, resolution=self.n_x_cond, low_percentile=0, high_percentile=100)
    grid_y = get_variable_grid(self.Y, resolution=int(y_res ** (1/self.Y.shape[1])), low_percentile=0, high_percentile=100)

    X, Y = cartesian_along_axis_0(grid_x, grid_y)

    # squeeze: in scipy.stats.entropy, Z_P of shape (n,1) yields a result of shape (n,) instead of a scalar
    Z_P = np.squeeze(P(X,Y))

    t_start = time.time()
    Z_Q = np.squeeze(Q(X,Y))


    time_to_predict = (time.time() - t_start) * 1000 / X.shape[0]  # time to predict per 1000


    if measure_time:
      return np.nan_to_num(scipy.stats.entropy(pk=Z_P, qk=Z_Q)), time_to_predict
    else:
      return np.nan_to_num(scipy.stats.entropy(pk=Z_P, qk=Z_Q))

  def kl_divergence_mc(self, x_cond, n_samples=10 ** 7, batch_size=None):
    """ Computes the Kullback–Leibler divergence via monte carlo integration using importance sampling with a cauchy distribution

    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      KL divergence of each x value to condition on - numpy array of shape (n_values,)
    """
    np.random.seed(self.seed)
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def kl_div(samples, x):
      p = P(x, samples)
      q = Q(x, samples)
      q = np.ma.masked_where(q < 10 ** -64, q)
      p = np.ma.masked_where(p < 10 ** -64, p)

      r = p * np.log(p / q)
      return r.filled(0)

    distances = self._mc_integration_cauchy(kl_div, x_cond, n_samples=n_samples, batch_size=batch_size)

    assert distances.ndim == 1 and distances.shape[0] == x_cond.shape[0]
    return distances

  def js_divergence_mc(self, x_cond, n_samples=10 ** 7, batch_size=None):
    """ Computes the Jason Shannon divergence via monte carlo integration using importance sampling with a cauchy distribution
    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      jason-shannon divergence of each x value to condition on - numpy array of shape (n_values,)
    """
    np.random.seed(self.seed)
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def js_div(samples, x):
      p = P(x, samples)
      q = Q(x, samples)
      q = np.ma.masked_where(q < 10 ** -64, q)
      p = np.ma.masked_where(p < 10 ** -64, p)
      m = 0.5 * (p + q)
      r = 0.5 * ((p * np.log(p / m)) + (q * np.log(q / m)))

      return r.filled(0)

    distances = self._mc_integration_cauchy(js_div, x_cond, n_samples=n_samples, batch_size=batch_size)

    assert distances.ndim == 1 and distances.shape[0] == x_cond.shape[0]
    return distances

  def hellinger_distance_mc(self, x_cond, n_samples=10 ** 7, batch_size=None):
    """ Computes the hellinger distance via monte carlo integration using importance sampling with a cauchy distribution

    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      squared hellinger distance of each x value to condition on - numpy array of shape (n_values,)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def hellinger_dist(samples, x):
      p = np.sqrt(P(x, samples))
      q = np.sqrt(Q(x, samples))
      r = (p - q)**2
      return r

    mc_integral = self._mc_integration_cauchy(hellinger_dist, x_cond, n_samples=n_samples, batch_size=batch_size)
    distances = np.sqrt(mc_integral / 2)

    assert distances.ndim == 1 and distances.shape[0] == x_cond.shape[0]
    return distances

  def _mc_integration_cauchy(self, func, x_cond, n_samples=10 ** 7, batch_size=None):
    if x_cond.shape[0] != n_samples:
      n_samples = x_cond.shape[0]
      logging.warning("number of samples reduced to %d since axis 0 of x_cond and samples must be equal.", x_cond.shape[0])


    if batch_size is None:
      n_batches = 1
      batch_size = n_samples
    else:
      n_batches = n_samples // batch_size + int(n_samples % batch_size > 0)

    distances = np.zeros(x_cond.shape[0])

    for i in range(x_cond.shape[0]):  # iterate over x values to condition on
      batch_result = np.zeros(n_batches)
      for j in range(n_batches):
        samples = stats.cauchy.rvs(loc=0, scale=2, size=(batch_size, self.estimator.ndim_x))
        f = _multidim_cauchy_pdf(samples, loc=0, scale=2)
        r = func(samples, x_cond)
        batch_result[j] = np.mean(r / f)
      distances[i] = batch_result.mean()

    assert distances.ndim == 1 and distances.shape[0] == x_cond.shape[0]
    return distances

  def compute_results(self):
    """
    Computes the statistics and returns a GoodnessOfFitResults object
    :return: GoodnessOfFitResults object that holds the computed statistics
    """
    x_cond = get_variable_grid(self.X, resolution=int(self.n_x_cond ** (1/self.X.shape[1])))

    gof_result = GoodnessOfFitResults(x_cond, self.estimator, self.probabilistic_model)

    # KL - Divergence
    gof_result.kl_divergence, gof_result.time_to_predict = self.kl_divergence(measure_time=True)

    # Hellinger distance
    gof_result.hellinger_distance = self.hellinger_distance_mc(x_cond=x_cond)

    # Kolmogorov Smirnov
    if self.estimator.ndim_y == 1 and self.estimator.can_sample:
      for i in range(x_cond.shape[0]):
        gof_result.ks_stat[i], gof_result.ks_pval[i] = self.kolmogorov_smirnov_cdf(x_cond[i, :])

    # Add time measurement
    gof_result.time_to_fit = self.time_to_fit

    # Add number of observattions
    gof_result.n_observations = self.n_observations

    gof_result.compute_means()
    return gof_result


  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}".format(
      self.estimator, self.probabilistic_model, self.n_observations, self.n_x_cond))


def get_variable_grid(X, resolution=20, low_percentile = 10, high_percentile=90):
  """
  computes grid of equidistant points between the specified percentiles in X
  :param X: data on which the percentiles shall be computed - ndarray with shape (n_samples, ndim_x)
  :param resolution: number of equidistant points in each direction
  :param low_percentile: lower percentile (int)
  :param high_percentile: upper percentile (int)
  :return: ndarray of shape (resolution * ndim_x, ndim_x)
  """
  assert 0 <= low_percentile < high_percentile <= 100

  if X.ndim == 1:
    X = np.expand_dims(X, axis=1)

  linspaces = []
  for i in range(X.shape[1]):
    low = np.percentile(X[:,i], low_percentile)
    high = np.percentile(X[:,i], high_percentile)
    linspaces.append(np.linspace(low,high, num=resolution))

  grid = np.vstack([grid_dim.flatten() for grid_dim in np.meshgrid(*linspaces)]).T

  assert grid.shape[1] == X.shape[1]
  return grid

def cartesian_along_axis_0(X, Y):
  """
  calculates the cartesian product of two matrices along the axis 0
  :param X: ndarray of shape (x_shape_0, ndim_x)
  :param Y: ndarray of shape (y_shape_0, ndim_y)
  :return: (X_res, Y_res) of shape (x_shape_0 * y_shape_0, ndim_x) resp. (x_shape_0 * y_shape_0, ndim_y)
  """
  assert X.ndim == Y.ndim == 2
  target_len = X.shape[0] * Y.shape[0]
  X_res = np.zeros(shape=(target_len, X.shape[1]))
  Y_res = np.zeros(shape=(target_len, Y.shape[1]))
  k = 0
  for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
      X_res[k, :] = X[i, :]
      Y_res[k, :] = Y[j, :]
  return X_res, Y_res

def _multidim_cauchy_pdf(x, loc=0, scale=2):
  """ multidimensional cauchy pdf """

  return np.exp(_multidim_cauchy_log_pdf(x, loc=loc, scale=scale))

def _multidim_cauchy_log_pdf(x, loc=0, scale=2):
  """ multidimensional cauchy log pdf """
  assert x.ndim == 2

  p = np.zeros(x.shape[0])
  for i in range(x.shape[1]):
    p += stats.cauchy(loc=0, scale=scale).logpdf(x[:, i])
  return p
