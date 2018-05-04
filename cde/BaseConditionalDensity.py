from sklearn.base import BaseEstimator
import numpy as np
from .helpers import *
import scipy.stats as stats

class ConditionalDensity(BaseEstimator):

  def _mean_mc(self, x_cond, n_samples=10 ** 7):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, samples = sample(x)
      means[i, :] = np.mean(samples, axis=0)
    return means

  def _mean_pdf(self, x_cond, n_samples=10 ** 7):
    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      func = lambda y: y * np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y))
      integral = mc_integration_cauchy(func, ndim=2, n_samples=n_samples)
      means[i] = integral
    return means

  def _covariance_pdf(self, x_cond, n_samples=10 ** 6):
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    mean = self.mean_(x_cond)
    for i in range(x_cond.shape[0]):
      x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))

      def cov(y):
        a = (y - mean[i])

        # compute cov matrices c for sampled instances and weight them with the probability p from the pdf
        c = np.empty((a.shape[0], a.shape[1] ** 2))
        for j in range(a.shape[0]):
          c[j, :] = np.outer(a[j], a[j]).flatten()

        p = np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y ** 2))
        res = c * p
        return res

      integral = mc_integration_cauchy(cov, ndim=self.ndim_y, n_samples=n_samples)
      covs[i] = integral.reshape((self.ndim_y, self.ndim_y))
    return covs

  def _covariance_mc(self, x_cond, n_samples=10 ** 7):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, y_sample = sample(x)

      c = np.cov(y_sample, rowvar=False)
      covs[i] = c
    return covs

  def _quantile_mc(self, x_cond, alpha=0.01, n_samples=10 ** 7):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    assert x_cond.ndim == 2
    VaRs = np.zeros(x_cond.shape[0])
    x_cond = np.tile(x_cond.reshape((1, x_cond.shape[0], x_cond.shape[1])), (n_samples,1, 1))
    for i in range(x_cond.shape[1]):
      _, samples = sample(x_cond[:, i,:])
      VaRs[i] = np.percentile(samples, alpha * 100.0)
    return VaRs

  def _quantile_cdf_old(self, x_cond, alpha=0.01, eps=10 ** -8):
    # older implementation of Newton Method for finding quantiles
    approx_error = 10 ** 8
    left, right = -10 ** 8, 10 ** 8

    VaRs = np.zeros(x_cond.shape[0])

    # numerical approximation, i.e. Newton method
    for j in range(x_cond.shape[0]):
      while approx_error > eps:
        middle = (left + right) / 2
        y = np.array([middle])

        p = self.cdf(x_cond[j, :], y)

        if p > alpha:
          right = middle
        else:
          left = middle
        approx_error = np.abs(p - alpha)
      VaRs[j] = middle
    return VaRs

  def _quantile_cdf(self, x_cond, alpha=0.01, eps=10 ** -8, start_value=-0.1):
    # Newton Method for finding the alpha quantile of a conditional distribution
    approx_error = 10 ** 8
    q = start_value

    VaRs = np.zeros(x_cond.shape[0])
    for j in range(x_cond.shape[0]):
      while approx_error > eps:
        F = self.cdf(x_cond[j, :], np.array([q]))
        f = self.pdf(x_cond[j, :], np.array([q]))
        q = q - (F - alpha)/ f

        approx_error = np.abs(F - alpha)
      VaRs[j] = q
    return VaRs

  def _conditional_value_at_risk_mc_pdf(self, VaRs, x_cond, alpha=0.01, n_samples=10 ** 7):
    assert VaRs.shape[0] == x_cond.shape[0], "same number of x_cond must match the number of values_at_risk provided"

    CVaRs = np.zeros(x_cond.shape[0])

    # preparations for importance sampling from exponential distribtution
    scale = 0.4 # 1 \ lambda
    sampling_dist = stats.expon(scale=scale)
    exp_samples = sampling_dist.rvs(size=n_samples).flatten()
    exp_f = sampling_dist.pdf(exp_samples)  #1 / scale * np.exp(-exp_samples/scale)

    # check shapes
    assert exp_samples.shape[0] == exp_f.shape[0] == n_samples

    for i in range(x_cond.shape[0]):
      # flip the normal exponential distribution by negating it & placing it's mode at the VaR value
      y_samples = VaRs[i] - exp_samples

      x_cond_tiled = np.tile(np.expand_dims(x_cond[i], axis=0), (n_samples, self.ndim_x))
      assert x_cond_tiled.shape == (n_samples, self.ndim_x)

      p = self.pdf(x_cond_tiled, y_samples).flatten()
      q = exp_f.flatten()
      importance_weights = p / q
      cvar = np.mean(y_samples * importance_weights, axis=0) / alpha
      CVaRs[i] = cvar

    return CVaRs

  def _conditional_value_at_risk_sampling(self, VaRs, x_cond, n_samples=10 ** 7):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    CVaRs = np.zeros(x_cond.shape[0])
    x_cond = np.tile(x_cond.reshape((1, x_cond.shape[0], x_cond.shape[1])), (n_samples, 1, 1))
    for i in range(x_cond.shape[1]):
      _, samples = sample(x_cond[:, i, :])
      shortfall_samples = np.ma.masked_where(VaRs[i] < samples, samples)
      CVaRs[i] = np.mean(shortfall_samples)

    return CVaRs

  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)

    if Y is not None:
      if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

      assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
      assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    if fitting:  # store n_dim of training data
      self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    else:
      assert X.shape[1] == self.ndim_x, "X must have shape (?, %i) but provided X has shape %s" % (self.ndim_x, X.shape)
      if Y is not None:
        assert Y.shape[1] == self.ndim_y, "Y must have shape (?, %i) but provided Y has shape %s" % (
        self.ndim_y, Y.shape)

    if Y is None:
      return X
    else:
      return X, Y

