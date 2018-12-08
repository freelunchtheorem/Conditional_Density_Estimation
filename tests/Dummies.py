import sys
import os
import scipy.stats as stats
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.density_simulation import BaseConditionalDensitySimulation
from cde.density_estimator import BaseDensityEstimator



class GaussianDummy(BaseDensityEstimator):

  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
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
    assert self.cov.shape[0] == self.cov.shape[1] == self.ndim_y

    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

  def fit(self, X, Y, verbose=False):
    self.fitted = True

  def pdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.gaussian.cdf(Y)

  def sample(self, X):
    X = self._handle_input_dimensionality(X)
    if np.size(X) == 1:
      Y = self.gaussian.rvs(size=1)
    else:
      Y = self.gaussian.rvs(size=(X.shape[0]))
    return X,Y

  def __str__(self):
    return str('\nEstimator type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n' .format(self.__class__.__name__, self.ndim_x, self.ndim_y, self.mean))


class SkewNormalDummy(BaseDensityEstimator):

  def __init__(self, shape=1, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    assert ndim_y == 1, "only on-dimensional y supported for skew normal dummy"

    self.shape = shape

    self.distribution = stats.skewnorm(a=shape)
    self.fitted = False

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

  def fit(self, X, Y, verbose=False):
    self.fitted = True

  def pdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.distribution.pdf(Y).flatten()

  def cdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.distribution.cdf(Y).flatten()

  def sample(self, X):
    X = self._handle_input_dimensionality(X)
    if np.size(X) == 1:
      Y = self.distribution.rvs(size=1)
    else:
      Y = self.distribution.rvs(size=(X.shape[0]))
    return X,Y

  @property
  def skewness(self):
    gamma = self.shape / np.sqrt(1+self.shape**2)
    skew = ((4-np.pi) / 2) * ((gamma * np.sqrt(2/np.pi))**3 / (1 - 2 * gamma**2 / np.pi )**(3/2))
    return skew

  @property
  def kurtosis(self):
    gamma = self.shape / np.sqrt(1 + self.shape ** 2)
    kurt = 2*(np.pi - 3) * (gamma * np.sqrt(2/np.pi))**4 / (1 - 2*gamma**2 / np.pi)**2
    return kurt


class SimulationDummy(BaseConditionalDensitySimulation):
  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
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

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

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
    Y = self.gaussian.rvs(size=X.shape[0])
    return X, Y

  def __str__(self):
    return str('\nProbabilistic model type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n cov: {}\n'.format(self.__class__.__name__, self.ndim_x,
                self.ndim_y, self.mean, self.cov))
