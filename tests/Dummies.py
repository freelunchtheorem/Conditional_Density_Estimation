from density_simulation import ConditionalDensity
from density_estimator import BaseDensityEstimator
import scipy.stats as stats
import numpy as np


class GaussianDummy(BaseDensityEstimator):

  def __init__(self, mean):
    self.ndim_x = 1
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = np.array(self.ndim_y*[mean])
    self.cov = np.identity(self.ndim_y)
    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False

  def fit(self, X, Y):
    self.fitted = True

  def pdf(self, X, Y):
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    return self.gaussian.cdf(Y)

  def sample(self, X):
    Y = self.gaussian.rvs(size=(X.shape[0]))
    return X, Y


class SimulationDummy(ConditionalDensity):
  def __init__(self, mean):
    self.ndim_x = 1
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = np.array(self.ndim_y*[mean])
    self.cov = np.identity(self.ndim_y)
    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False

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