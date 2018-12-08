import time
import numpy as np
import scipy
import logging
import matplotlib.pyplot as plt
from ml_logger import logger



from scipy.stats import shapiro, kstest
from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cde.density_simulation import BaseConditionalDensitySimulation
from cde.evaluation.GoodnessOfFitSingleResult import GoodnessOfFitSingleResult
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
    seed: random seed to draw samples from the probabilistic model
    task_name: specifies a unique name fo the GoodnessOfFit run, e.g. KernelMixtureNetwork_task_19. If task_name was not set during call,
    the name was specified in the estimator object (estimator.name) is used. If it was not specified there either, it is set to
     estimator and prob_model name.

  """
  def __init__(self, estimator, probabilistic_model, X, Y, n_observations, x_cond, n_mc_samples, seed=24, task_name=None):

    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, BaseConditionalDensitySimulation), "probabilistic model must inherit from ConditionalDensity"

    np.seterr(divide='ignore')

    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations
    self.x_cond = x_cond
    self.n_mc_samples = n_mc_samples

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    self.seed = seed
    np.random.seed(seed)
    self.X = X
    self.Y = Y
    self.n_observations = n_observations

    self.estimator = estimator

    if task_name is not None:
      self.task_name = task_name
    elif hasattr(self.estimator, 'name'):
      self.task_name = str(self.estimator.name)
    else:
      self.task_name = type(self.estimator).__name__ + '_' + type(self.probabilistic_model).__name__

  def fit_estimator(self, print_fit_result=True): #todo set to False
    """
    Fits the estimator with the provided data

    Args:
      print_fit_result: boolean that specifies whether the fitted distribution shall be plotted (only works if ndim_x and ndim_y = 1)
    """

    self.time_to_fit = None
    if not self.estimator.fitted:  # fit estimator if necessary
      t_start = time.time()
      self.estimator.fit(self.X, self.Y, verbose=False)
      self.time_to_fit = (time.time() - t_start) * self.n_observations / 1000  # time to fit per 1000 samples

    if print_fit_result and self.estimator.fitted:
      if self.probabilistic_model.ndim_x == 1 and self.probabilistic_model.ndim_y == 1:
        plt3d_true = self.probabilistic_model.plot(mode="pdf", numpyfig=False)
        logger.log_pyplot(key=self.task_name, fig=plt3d_true)
        plt.close(plt3d_true)

      if self.estimator.ndim_x == 1 and self.estimator.ndim_y == 1:
        plt2d = self.estimator.plot2d(show=False, numpyfig=False)
        plt3d = self.estimator.plot3d(show=False, numpyfig=False)
        logger.log_pyplot(key=self.task_name + "_fitted_cond_distr_2d", fig=plt2d)
        logger.log_pyplot(key=self.task_name + "_fitted_cond_distr_3d", fig=plt3d)
        plt.close(plt2d)
        plt.close(plt3d)

  def kolmogorov_smirnov_cdf(self, x_cond, n_samples=10**6):
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
    grid_x = get_variable_grid(self.X, resolution=self.x_cond, low_percentile=0, high_percentile=100)
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

  def kl_divergence_mc(self, n_samples=10**7, batch_size=None):
    """ Computes the Kullback–Leibler divergence via monte carlo integration using importance sampling with a cauchy distribution

    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      KL divergence of each x value to condition on - numpy array of shape (n_values,)
    """
    np.random.seed(self.seed)
    assert self.x_cond.ndim == 2 and self.x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def kl_div(samples, x):
      p = P(x, samples).flatten()
      q = Q(x, samples).flatten()
      q = np.ma.masked_where(q < 10 ** -64, q).flatten()
      p = np.ma.masked_where(p < 10 ** -64, p).flatten()

      r = p * np.log(p / q)
      return r.filled(0)

    distances = self._mc_integration_cauchy(kl_div, n_samples=n_samples, batch_size=batch_size)

    assert distances.ndim == 1 and distances.shape[0] == self.x_cond.shape[0]
    return distances

  def js_divergence_mc(self, n_samples=10**7, batch_size=None):
    """ Computes the Jason Shannon divergence via monte carlo integration using importance sampling with a cauchy distribution
    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      jason-shannon divergence of each x value to condition on - numpy array of shape (n_values,)
    """
    np.random.seed(self.seed)
    assert self.x_cond.ndim == 2 and self.x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def js_div(samples, x):
      p = P(x, samples).flatten()
      q = Q(x, samples).flatten()
      q = np.ma.masked_where(q < 10 ** -64, q)
      p = np.ma.masked_where(p < 10 ** -64, p)
      m = 0.5 * (p + q)
      r = 0.5 * ((p * np.log(p / m)) + (q * np.log(q / m)))

      return r.filled(0)

    distances = self._mc_integration_cauchy(js_div, n_samples=n_samples, batch_size=batch_size)

    assert distances.ndim == 1 and distances.shape[0] == self.x_cond.shape[0]
    return distances

  def hellinger_distance_mc(self, n_samples=10**7, batch_size=None):
    """ Computes the hellinger distance via monte carlo integration using importance sampling with a cauchy distribution

    Args:
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space
     batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

    Returns:
      hellinger distance of each x value to condition on - numpy array of shape (n_values,)
    """
    assert self.x_cond.ndim == 2 and self.x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    def hellinger_dist(samples, x):
      p = np.sqrt(P(x, samples)).flatten()
      q = np.sqrt(Q(x, samples)).flatten()
      r = (p - q)**2
      return r

    mc_integral = self._mc_integration_cauchy(hellinger_dist, n_samples=n_samples, batch_size=batch_size)
    distances = np.sqrt(mc_integral / 2)

    assert distances.ndim == 1 and distances.shape[0] == self.x_cond.shape[0]
    return distances

  def divergence_measures_pdf(self, n_samples=10**7):
    """ Computes kl-divergence, js-divergence and hellinger distance

      Args:
       n_samples: number of samples for monte carlo integration over the y space

      Returns:
        - hellinger distance of each x value to condition on - numpy array of shape (n_values,)
        - KL divergence of each x value to condition on - numpy array of shape (n_values,)
        - jason-shannon divergence of each x value to condition on - numpy array of shape (n_values,)
      """
    assert self.x_cond.ndim == 2 and self.x_cond.shape[1] == self.estimator.ndim_x

    P = self.probabilistic_model.pdf
    Q = self.estimator.pdf

    hell_distances = np.zeros(self.x_cond.shape[0])
    kl_divs = np.zeros(self.x_cond.shape[0])
    js_divs = np.zeros(self.x_cond.shape[0])

    batch_size = int(n_samples)

    for i in range(self.x_cond.shape[0]):
      samples = stats.cauchy.rvs(loc=0, scale=1, size=(batch_size, self.estimator.ndim_x))
      f = _multidim_cauchy_pdf(samples, loc=0, scale=1).flatten()
      x = np.tile(self.x_cond[i].reshape((1, self.x_cond[i].shape[0])), (samples.shape[0], 1))

      p = P(x, samples).flatten()
      q = Q(x, samples).flatten()

      hell_distances[i] = np.mean(_hellinger_dist(p, q).flatten() / f)
      kl_divs[i] = np.mean(_kl_divergence(p, q).flatten() / f)
      js_divs[i] = np.mean(_js_divergence(p, q).flatten() / f)

    # extra treatment for hellinger
    hell_distances = np.sqrt(hell_distances / 2)

    assert hell_distances.ndim == 1 and hell_distances.shape[0] == self.x_cond.shape[0]
    assert kl_divs.ndim == 1 and kl_divs.shape[0] == self.x_cond.shape[0]
    assert js_divs.ndim == 1 and js_divs.shape[0] == self.x_cond.shape[0]

    return hell_distances, kl_divs, js_divs

  def wasserstein_distance_mc(self, x_cond, n_samples=10**7, batch_size=None):
    """ Computes the Wasserstein distance via monte carlo integration using importance sampling with a cauchy distribution

      Args:
       x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
       n_samples: number of samples for monte carlo integration over the y space
       batch_size: (optional) batch size for computing mc estimates - if None: batch_size is set to n_samples

      Returns:
        wasserstein distance for each x value to condition on - numpy array of shape (n_values,)
      """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.estimator.ndim_x
    assert hasattr(self.probabilistic_model, "cdf")
    assert hasattr(self.estimator, "cdf")

    raise NotImplementedError("Wasserstein distance MC doesn't produce reliable results")

    P = self.probabilistic_model.cdf
    Q = self.estimator.cdf

    def wasserstein_dist(samples, x):
      return np.abs(P(x, samples) - Q(x, samples))

      #return stats.multivariate_normal.pdf(samples, mean=[0,0], cov=np.diag([1,1]))

    distances = self._mc_integration_cauchy(wasserstein_dist, x_cond, n_samples=n_samples, batch_size=batch_size)

    assert distances.ndim == 1 and distances.shape[0] == x_cond.shape[0]
    return distances

  def _mc_integration_cauchy(self, func, n_samples=10 ** 7, batch_size=None):
    if batch_size is None:
      n_batches = 1
      batch_size = n_samples
    else:
      n_batches = n_samples // batch_size + int(n_samples % batch_size > 0)

    distances = np.zeros(self.x_cond.shape[0])

    for i in range(self.x_cond.shape[0]):  # iterate over x values to condition on
      batch_result = np.zeros(n_batches)
      for j in range(n_batches):
        samples = stats.cauchy.rvs(loc=0, scale=2, size=(batch_size, self.estimator.ndim_x))
        f = _multidim_cauchy_pdf(samples, loc=0, scale=2)
        x = np.tile(self.x_cond[i].reshape((1, self.x_cond[i].shape[0])), (samples.shape[0],1))
        r = func(samples, x)
        r, f = r.flatten(), f.flatten() # flatten r to avoid strange broadcasting behavior
        batch_result[j] = np.mean(r / f)
      distances[i] = batch_result.mean()

    assert distances.ndim == 1 and distances.shape[0] == self.x_cond.shape[0]
    return distances

  def compute_results(self):
    """
      Computes statistics and stores the results in GoodnessOfFitResult object

      Returns:
        GoodnessOfFitResult object that holds the computed statistics
    """
    assert self.x_cond.all()
    assert self.estimator is not None
    assert self.probabilistic_model is not None

    gof_result = GoodnessOfFitSingleResult(self.x_cond, self.estimator.get_configuration(), self.probabilistic_model.get_configuration())

    if self.n_mc_samples < 10**5:
      logging.warning("using less than 10**5 samples for monte carlo not recommended")

    # Divergence Measures
    gof_result.hellinger_distance_, gof_result.kl_divergence_, gof_result.js_divergence_ = self.divergence_measures_pdf(n_samples=self.n_mc_samples)

    gof_result.hellinger_distance = [np.mean(gof_result.hellinger_distance_)]
    gof_result.kl_divergence = [np.mean(gof_result.kl_divergence_)]
    gof_result.js_divergence = [np.mean(gof_result.js_divergence_)]

    # Add number of observations
    gof_result.n_observations = [self.n_observations]

    gof_result.x_cond = [str(self.x_cond.flatten())]

    gof_result.x_cond_ = self.x_cond # original data preserved

    gof_result.n_mc_samples = [self.n_mc_samples]

    if self.estimator.can_sample:
      """ create strings since pandas requires lists to be all of the same length if numerical """
      gof_result.mean_est_ = self.estimator.mean_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
      gof_result.mean_est = [str(gof_result.mean_est_.flatten())]

      gof_result.cov_est_ = self.estimator.covariance(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
      gof_result.cov_est = [str(gof_result.cov_est_.flatten())]

      gof_result.mean_sim_ = self.probabilistic_model.mean_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
      gof_result.mean_sim = [str(gof_result.mean_sim_ .flatten())]


      gof_result.cov_sim_ = self.probabilistic_model.covariance(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
      gof_result.cov_sim = [str(gof_result.cov_sim_.flatten())]

      """ absolute mean, cov difference """
      gof_result.mean_abs_diff = np.mean(np.abs(gof_result.mean_est_ - gof_result.mean_sim_))
      gof_result.cov_abs_diff = np.mean(np.abs(gof_result.cov_est_ - gof_result.cov_sim_))

    """ tail risk """
    if self.estimator.ndim_y == 1:

      gof_result.VaR_sim_, gof_result.CVaR_sim_ = self.probabilistic_model.tail_risk_measures(self.x_cond, n_samples=self.n_mc_samples)
      gof_result.VaR_sim = [str(gof_result.VaR_sim_.flatten())]
      gof_result.CVaR_sim = [str(gof_result.CVaR_sim_.flatten())]

      gof_result.VaR_est_, gof_result.CVaR_est_ = self.estimator.tail_risk_measures(self.x_cond, n_samples=self.n_mc_samples)
      gof_result.VaR_est = [str(gof_result.VaR_est_.flatten())]
      gof_result.CVaR_est = [str(gof_result.CVaR_est_.flatten())]

      gof_result.VaR_abs_diff = np.mean(np.abs(gof_result.VaR_sim_ - gof_result.VaR_est_))
      gof_result.CVaR_abs_diff = np.mean(np.abs(gof_result.CVaR_sim_ - gof_result.CVaR_est_))

    """ time to fit """
    gof_result.time_to_fit = self.time_to_fit

    return gof_result

  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}".format(
      self.estimator, self.probabilistic_model, self.n_observations, self.x_cond))

def _hellinger_dist(p, q):
  assert p.shape == q.shape
  q = np.ma.masked_where(q < 0.0, q).flatten()
  p = np.ma.masked_where(p < 0.0, p).flatten()
  return (np.sqrt(p) - np.sqrt(q)) ** 2

def _kl_divergence(p, q):
  assert p.shape == q.shape
  q = np.ma.masked_where(q < 10 ** -64, q).flatten()
  p = np.ma.masked_where(p < 10 ** -64, p).flatten()
  kl = p * np.log(p / q)
  return kl.filled(0)

def _js_divergence(p, q):
  assert p.shape == q.shape
  q = np.ma.masked_where(q < 10 ** -64, q).flatten()
  p = np.ma.masked_where(p < 10 ** -64, p).flatten()
  m = 0.5 * (p + q)
  r = 0.5 * ((p * np.log(p / m)) + (q * np.log(q / m)))
  return r.filled(0)


def sample_x_cond(X, n_x_cond=20, low_percentile = 10, high_percentile=90, random_seed=92):
  """
  uniformly samples n_xcond points within the specified percentiles in X

  Args:
    X: data on which the percentiles shall be computed - ndarray with shape (n_samples, ndim_x)
    n_x_cond: number of x_cond points to be sampled
    low_percentile: lower percentile (int)
    high_percentile: upper percentile (int)

  Returns:
    sampled x_cond points - ndarray of shape (n_xcond, ndim_x)
  """
  assert 0 <= low_percentile < high_percentile <= 100
  rand = np.random.RandomState(random_seed)

  if X.ndim == 1:
    X = np.expand_dims(X, axis=1)

  samples_per_dim = []
  for i in range(X.shape[1]):
    low = np.percentile(X[:,i], low_percentile)
    high = np.percentile(X[:,i], high_percentile)
    samples_per_dim.append(rand.uniform(low, high, size=(n_x_cond, 1)))

  x_cond = np.hstack(samples_per_dim)

  assert x_cond.shape[1] == X.shape[1] and x_cond.shape[0] == n_x_cond
  return x_cond

def get_variable_grid(X, resolution=20, low_percentile = 10, high_percentile=90):
  """
  computes grid of equidistant points between the specified percentiles in X
  Args:
    X: data on which the percentiles shall be computed - ndarray with shape (n_samples, ndim_x)
    resolution: number of equidistant points in each direction
    low_percentile: lower percentile (int)
    high_percentile: upper percentile (int)
  Returns:
    ndarray of shape (resolution * ndim_x, ndim_x)
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

  p = stats.cauchy.pdf(x, loc=loc, scale=scale)
  p = np.prod(p, axis=1).flatten()
  assert p.ndim == 1
  return p
