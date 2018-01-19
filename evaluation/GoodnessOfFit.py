from scipy.stats import shapiro, kstest, jarque_bera, ks_2samp
from density_estimator.base import BaseDensityEstimator
from density_simulation import ConditionalDensity
from joblib import Parallel, delayed
import numpy as np
import scipy
import time


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


  def kolmogorov_smirnov_cdf(self, x_cond, n_samples=1000):
    """
    Calculates Kolmogorov-Smirnov Statistics
    :param x_cond: x value to condition on
    :param n_samples: number of samples y drawn from p(y|x_cond)
    :return: (ks_stat, ks_pval) Kolmogorov-Smirnov statistic and p-pvalue
    """
    X_cond = np.vstack([x_cond for _ in range(n_samples)])
    np.random.seed(self.seed**2)
    _, estimator_conditional_samples = self.estimator.sample(X_cond)
    assert estimator_conditional_samples.ndim == 1 or estimator_conditional_samples.shape[1] == 1 , "Can only compute Kosmogorov Smirnov Statistic of ndim_y = 1"
    estimator_conditional_samples = estimator_conditional_samples.flatten()
    return kstest(estimator_conditional_samples, lambda y: self.probabilistic_model.cdf(X_cond, y))

  def kl_divergence(self, y_res=100, measure_time = False):
    """
    Calculates the discrete approximiation of KL_divergence of the fitted ditribution w.r.t. the true distribution
    :param y_res: sampling rate for y
    :param measure_time: boolean that indicates whether the time_to_predict 1000 density function values shall be returned as second element
    :return: Kl-divergence (float), time_to_predict (float) if measure_time is true
    """
    P = self.probabilistic_model.pdf
    Q = self.estimator.predict

    # prepare mesh
    grid_x = get_variable_grid(self.X, resolution=self.n_x_cond, low_percentile=0, high_percentile=100)
    grid_y = get_variable_grid(self.Y, resolution=int(y_res ** (1/self.Y.shape[1])), low_percentile=0, high_percentile=100)

    X, Y = cartesian_along_axis_0(grid_x, grid_y)


    Z_P = P(X,Y)

    t_start = time.time()
    Z_Q = Q(X,Y)
    time_to_predict = (time.time() - t_start) * 1000 / X.shape[0]  # time to predict per 1000

    if measure_time:
      return scipy.stats.entropy(pk=Z_P, qk=Z_Q), time_to_predict
    else:
      return scipy.stats.entropy(pk=Z_P, qk=Z_Q)

  def compute_results(self):
    """
    Computes the statistics and returns a GoodnessOfFitResults object
    :return: GoodnessOfFitResults object that holds the computed statistics
    """
    x_cond = get_variable_grid(self.X, resolution=int(self.n_x_cond ** (1/self.X.shape[1])))

    gof_result = GoodnessOfFitResults(x_cond, self.estimator, self.probabilistic_model)

    # KL - Divergence
    gof_result.mean_kl, gof_result.time_to_predict = self.kl_divergence(measure_time=True)

    # Kolmogorov Smirnov
    if self.estimator.ndim_y == 1 and self.estimator.can_sample:
      print(x_cond.shape[0])
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

class GoodnessOfFitResults:
  def __init__(self, x_cond, estimator, probabilistic_model):
    self.cond_values = x_cond

    self.time_to_fit = None
    self.time_to_predict = None

    self.ndim_x = estimator.ndim_x
    self.ndim_y = estimator.ndim_y

    self.estimator_params = estimator.get_params()
    self.probabilistic_model_params = probabilistic_model.get_params()

    if estimator.ndim_y > 1: # cannot perform KS Test -> net respective variables to None
      self.ks_stat = None
      self.ks_pval = None
    else:
      self.ks_stat = np.zeros(x_cond.shape[0])
      self.ks_pval = np.zeros(x_cond.shape[0])


    self.mean_kl = None
    self.mean_ks_stat = None
    self.mean_ks_pval = None

  def compute_means(self):
    if self.ks_stat is not None and self.ks_pval is not None:
      self.mean_ks_stat = self.ks_stat.mean()
      self.mean_ks_pval = self.ks_pval.mean()

  def report_dict(self):
    full_dict = self.__dict__
    keys_of_interest = ["n_observations", "ndim_x", "ndim_y", "mean_kl", "mean_ks_stat", "mean_ks_pval", "time_to_fit", "time_to_predict"]
    report_dict = dict([(key, full_dict[key]) for key in keys_of_interest])

    get_from_dict = lambda key: self.estimator_params[key] if key in self.estimator_params else None

    for key in ["estimator", "n_centers", "center_sampling_method"]:
      report_dict[key] = get_from_dict(key)


    report_dict["simulator"] = self.probabilistic_model_params["simulator"]

    return report_dict




  def __str__(self):
    return "KL-Divergence: %.4f , KS Stat: %.4f, KS pval: %.4f"%(self.mean_kl, self.mean_ks_stat, self.mean_ks_pval)


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
