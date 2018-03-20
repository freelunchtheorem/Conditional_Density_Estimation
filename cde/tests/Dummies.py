from cde.density_simulation import ConditionalDensity
from cde.density_estimator import BaseDensityEstimator
import scipy.stats as stats
import numpy as np


class GaussianDummy(BaseDensityEstimator):

  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = mean
    # check if mean is scalar
    if isinstance(self.mean, list):
      self.mean = np.array(self.ndim_y * [self.mean])

    self.cov = cov

    if self.cov is None:
      self.cov = np.identity(self.ndim_y)


    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False
    self.can_sample = True

  def fit(self, X, Y):
    self.fitted = True

  def pdf(self, X, Y):
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    return self.gaussian.cdf(Y)

  def pdf(self, X, Y, resolution=50):
    return self.pdf(X,Y)

  def sample(self, X):
    if np.size(X) == 1:
      Y = self.gaussian.rvs(size=1)
    else:
      Y = self.gaussian.rvs(size=(X.shape[0]))
    return X,Y

  def __str__(self):
    return str('\nEstimator type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n' .format(self.__class__.__name__, self.ndim_x, self.ndim_y, self.mean))




class SimulationDummy(ConditionalDensity):
  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = mean
    self.cov = cov
    # check if mean is scalar
    if isinstance(self.mean, list):
      self.mean = np.array(self.ndim_y*[self.mean])

    if self.cov is None:
      self.cov = np.identity(self.ndim_y)

    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False
    self.can_sample = True

  def pdf(self, X, Y):
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    return self.gaussian.cdf(Y)

  def simulate(self, n_samples=1000):
    assert n_samples > 0
    X = self.gaussian.rvs(size=n_samples)
    Y = self.gaussian.rvs(size=n_samples)
    return X, Y

  def simulate_conditional(self, X):
    Y = self.gaussian.rvs(size=(X.shape[0]))
    return X, Y

  def __str__(self):
    return str('\nProbabilistic model type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n cov: {}\n'.format(self.__class__.__name__, self.ndim_x,
                self.ndim_y, self.mean, self.cov))
