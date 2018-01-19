from scipy.stats import shapiro, kstest, jarque_bera, ks_2samp
from density_estimator.base import BaseDensityEstimator
from density_simulation import ConditionalDensity
from joblib import Parallel, delayed
import numpy as np
import scipy


class GoodnessOfFit:
  """
  Class that takes an estimator a probabilistic simulation model. The estimator is fitted on n_obervation samples.
  Then the goodness of fit w.r.t to the true probability distribution is evaluated
  """
  def __init__(self, estimator, probabilistic_model, n_observations=100, n_x_cond=10, print_fit_result=False, seed=23):
    """
    :param estimator: Estimator instance that implements the functionality from BaseDensityEstimator (can be either fitted or not fitted)
    :param probabilistic_model: ConditionalDensity instance which implements the methods simulate, pdf, cdf
    :param n_observations: number of observations which are drawn from the probabilistic_model to fit the estimator.
    If the estimator is already fitted, this parameter is ignored
    :param n_x_cond: number of different x to condition on - e.g. if n_x_cond=20, p(y|x) is evaluated for 20 different x
    :param print_fit_result: boolean that specifies whether the fitted distribution shall be plotted (only works if ndim_x and ndim_y = 1)
    :param random seed to draw samples from the probabilistic model
    """

    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, ConditionalDensity), "probabilistic model must inherit from ConditionalDensity"

    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations
    self.n_x_cond = n_x_cond

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    np.random.seed(seed)
    self.X, self.Y = probabilistic_model.simulate(self.n_observations)

    if not estimator.fitted: #fit estimator if necessary
      estimator.fit(self.X, self.Y)

    if print_fit_result:
      self.probabilistic_model.plot(mode="pdf")
      estimator.plot()

    self.estimator = estimator


  def sample_conditional_values(self):
    _, estimator_conditional_samples = self.estimator.sample(self.X_cond)
    _, proba_model_conditional_samples = self.probabilistic_model.simulate_conditional(self.X_cond)

    """ kstest can't handle single-dimensional entries, therefore remove it"""
    if estimator_conditional_samples.ndim == 2:
      if estimator_conditional_samples.shape[1] == 1:
        estimator_conditional_samples = np.squeeze(estimator_conditional_samples, axis = 1)
    if proba_model_conditional_samples.ndim == 2:
      if proba_model_conditional_samples.shape[1] == 1:
        proba_model_conditional_samples = np.squeeze(proba_model_conditional_samples, axis=1)

    return estimator_conditional_samples, proba_model_conditional_samples

  def kolmogorov_smirnov_cdf(self, x_cond, n_samples=1000):
    X_cond = np.vstack([x_cond for _ in range(n_samples)])
    _, estimator_conditional_samples = self.estimator.sample(X_cond)
    assert estimator_conditional_samples.ndim == 1 or estimator_conditional_samples.shape[1] == 1 , "Can only compute Kosmogorov Smirnov Statistic of ndim_y = 1"
    estimator_conditional_samples = estimator_conditional_samples.flatten()
    return kstest(estimator_conditional_samples, lambda y: self.probabilistic_model.cdf(X_cond, y))


  def kolmogorov_smirnov_2sample(self):
    samples = np.asarray([self.sample_conditional_values() for _ in range(self.repeat_kolmogorov)])
    estimator_cond_samples = samples[:, 0]
    probabilistic_cond_samples = samples[:, 1]

    statistics = np.asarray(Parallel(n_jobs=-1)(delayed(ktest_2sample)(estimator_cond_samples[i], probabilistic_cond_samples[i]) for i in range(
      self.repeat_kolmogorov)))
    return np.mean(statistics[:,0]), np.mean(statistics[:,1])

  def kl_divergence(self, y_res=100):
    """
    Calculates the discrete approximiation of KL_divergence of the fitted ditribution w.r.t. the true distribution
    :param y_res: sampling rate for y
    :return: Kl-divergence (float)
    """
    P = self.probabilistic_model.pdf
    Q = self.estimator.predict

    # prepare mesh
    grid_x = get_variable_grid(self.X, resolution=self.n_x_cond, low_percentile=0, high_percentile=100)
    grid_y = get_variable_grid(self.Y, resolution=int(y_res ** (1/self.Y.shape[1])), low_percentile=0, high_percentile=100)

    X, Y = cartesian_along_axis_0(grid_x, grid_y)

    Z_P = P(X,Y)
    Z_Q = Q(X,Y)

    return scipy.stats.entropy(pk=Z_P, qk=Z_Q)

  def compute_results(self):
    x_cond = get_variable_grid(self.X, resolution=int(self.n_x_cond ** (1/self.X.shape[1])))

    gof_result = GoodnessOfFitResults(x_cond, self.estimator, self.probabilistic_model)

    # KL - Divergence
    gof_result.mean_kl = self.kl_divergence()

    # Kolmogorov Smirnov
    if self.estimator.ndim_y == 1:
      print(x_cond.shape[0])
      for i in range(x_cond.shape[0]):
        gof_result.ks_stat[i], gof_result.ks_pval[i] = self.kolmogorov_smirnov_cdf(x_cond[i, :])

    gof_result.compute_means()
    return gof_result


  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}\n repeat_kolmogorov: {}\n".format(self.estimator,
                                                                                                                 self.probabilistic_model,
                                                                                                                 self.n_observations,

                                                                                                                 self.n_x_cond, self.repeat_kolmogorov))


class GoodnessOfFitResults:

  def __init__(self, x_cond, estimator, probabilistic_model):
    self.cond_values = x_cond

    self.estimator_params = estimator.get_params()
    self.probabilistic_model_params = probabilistic_model.get_params()

    if estimator.ndim_y > 1: # cannot perform KS Test -> net respective variables to None
      self.ks_stat = None
      self.ks_pval = None
    else:
      self.ks_stat = np.zeros(x_cond.shape[0])
      self.ks_pval = np.zeros(x_cond.shape[0])

    self.kl = np.zeros(x_cond.shape[0])


    self.mean_kl = None
    self.mean_ks_stat = None
    self.mean_ks_pval = None

  def compute_means(self):
    if self.ks_stat is not None and self.ks_pval is not None:
      self.mean_ks_stat = self.ks_stat.mean()
      self.mean_ks_pval = self.ks_pval.mean()


  def __str__(self):
    return "KL-Divergence: %.4f , KS Stat: %.4f, KS pval: %.4f"%(self.mean_kl, self.mean_ks_stat, self.mean_ks_pval)




""" closured functions cannot be pickled -> required to be outside for parallel computing """
def ktest_cdf(cdf, X_cond, est_cond_samples):
  return kstest(est_cond_samples, lambda y: cdf(X_cond, y))


def ktest_2sample(estimator_cond_samples, probabilistic_cond_samples):
  return scipy.stats.ks_2samp(estimator_cond_samples, probabilistic_cond_samples)

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
  calculates the cartesian product of two matrixes along the axis 0
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
