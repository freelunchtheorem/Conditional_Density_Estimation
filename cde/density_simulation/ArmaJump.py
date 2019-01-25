import scipy.stats as stats
import numpy as np
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation


class ArmaJump(BaseConditionalDensitySimulation):
  """ AR(1) model with jump component

  Args:
    c: constant return component of AR(1)
    arma_a1: AR(1) factor
    std: standard deviation of the Gaussian Noise
    jump_prob: probability of a negative jump
    random_seed: seed for the random_number generator
  """

  def __init__(self, c=0.1, arma_a1=0.9, std=0.05, jump_prob=0.05, random_seed=None):
    self.std = std
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    # AR(1) params
    self.arma_c = c
    self.arma_a1 = arma_a1

    # Jump component
    assert jump_prob >= 0 and jump_prob <= 1
    self.jump_prob = jump_prob
    self.jump_mean = -3*c
    self.jump_std = 2*std

    self.ndim_x = 1
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    # approximate data statistics
    self.y_mean, self.y_std = self._compute_data_statistics()

    self.has_cdf = True
    self.has_pdf = True
    self.can_sample = True

  def pdf(self, X, Y):
    """ Conditional probability density function p(y|x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the pdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
      p(X|Y) conditional density values for the provided X and Y - numpy array of shape (n_points, )
    """
    X, Y = self._handle_input_dimensionality(X, Y)
    mean = self.arma_c * (1-self.arma_a1) + self.arma_a1 * X
    return (1-self.jump_prob) * stats.norm.pdf(Y - mean, scale=self.std).flatten() + \
           self.jump_prob * stats.norm.pdf(Y - (mean + self.jump_mean), scale=2*self.std).flatten()

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

        Args:
          X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
          Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

        Returns:
         P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
        """
    X, Y = self._handle_input_dimensionality(X, Y)
    mean = self.arma_c * (1 - self.arma_a1) + self.arma_a1 * X
    return (1-self.jump_prob) * stats.norm.cdf(Y - mean, scale=self.std).flatten() + \
           self.jump_prob * stats.norm.cdf(Y - (mean + self.jump_mean), scale=2*self.std).flatten()

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    mean = self.arma_c * (1 - self.arma_a1) + self.arma_a1 * X
    y_ar = self.random_state.normal(loc=mean, scale=self.std, size=X.shape[0])

    mean_jump = mean + self.jump_mean
    y_jump = self.random_state.normal(loc=mean_jump, scale=self.jump_std, size=X.shape[0])

    jump_bernoulli = self.random_state.uniform(size=X.shape[0]) < self.jump_prob

    return X, np.select([jump_bernoulli, np.bitwise_not(jump_bernoulli)], [y_jump, y_ar])

  def simulate(self, x_0=0, n_samples=1000, burn_in=100):
    """ Draws random samples from the unconditional distribution p(x,y)

       Args:
         n_samples: (int) number of samples to be drawn from the conditional distribution

       Returns:
         (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    self.eps = self.random_state.normal(scale=self.std, size=n_samples + burn_in + 1)

    x = np.zeros(n_samples + burn_in + 1)
    x[0] = x_0
    for i in range(1, n_samples + burn_in + 1):
      if self.random_state.uniform() > self.jump_prob: # AR(1)
        x[i] = self.arma_c * (1-self.arma_a1) + self.arma_a1 * x[i-1] + self.eps[i]
      else: # Jump
        jump = self.random_state.normal(loc=self.jump_mean, scale=self.jump_std)
        x[i] = self.arma_c * (1-self.arma_a1) + self.arma_a1 * x[i-1] + jump

    return x[burn_in:n_samples + burn_in], x[burn_in+1:n_samples + burn_in + 1]

  def mean_(self, x_cond, n_samples=None):
    """ Conditional mean of the distribution
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """

    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      mean = self.arma_c * (1 - self.arma_a1) + self.arma_a1 * x_cond[i]
      means[i, :] = self.jump_prob * (mean + self.jump_mean) + (1-self.jump_prob) * mean
    return means

  def covariance(self, x_cond, n_samples=None):
    """ Covariance of the distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """

    covs=np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    for i in range(x_cond.shape[0]):
      c1 = self.jump_prob * self.jump_std ** 2 + (1-self.jump_prob) * self.std ** 2

      mean = self.arma_c * (1 - self.arma_a1) + self.arma_a1 * x_cond[i]
      c2 = self.jump_prob * mean**2 + (1-self.jump_prob) * (mean-self.jump_mean)**2 - \
           (self.jump_prob * mean + (1-self.jump_prob) * (mean-self.jump_mean))**2
      covs[i][0][0] = c1 + c2

    return covs