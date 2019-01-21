import scipy.stats as stats
import numpy as np
from cde.density_simulation.BaseConditionalDensitySimulation import BaseConditionalDensitySimulation


class SkewNormal(BaseConditionalDensitySimulation):
  """ This model represents a univariate skewed normal distribution.

  """

  def __init__(self, random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    # parameters of the X to distribution parameters mapping
    self.loc_slope = 0.1
    self.loc_intercept = 0.0

    self.scale_square_param = 0.1
    self.scale_intercept = 0.05

    self.skew_low = -4
    self.skew_high = 0.0

    # x folows gaussian
    self.x_loc = 0
    self.x_scale = 0.5
    self.x_dist = stats.norm(loc=self.x_loc, scale=self.x_scale)

    self.ndim_x = 1
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    # approximate data statistics
    self.y_mean, self.y_std = self._compute_data_statistics()

    self.has_cdf = True
    self.has_pdf = True
    self.can_sample = True

  def _loc_scale_skew_mapping(self, X):
    loc = self.loc_intercept + self.loc_slope * X
    scale = self.scale_intercept + self.scale_square_param * X**2
    skew = self.skew_low + (self.skew_high - self.skew_low) * sigmoid(X)
    return loc, scale, skew

  def _sample_x(self, n_samples):
    return self.x_dist.rvs((n_samples,self.ndim_x), random_state=self.random_state)

  def pdf(self, X, Y):
    """ Conditional probability density function p(y|x) of the underlying probability model
(
    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the pdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
      p(X|Y) conditional density values for the provided X and Y - numpy array of shape (n_points, )
    """
    X, Y = self._handle_input_dimensionality(X, Y)

    locs, scales, skews = self._loc_scale_skew_mapping(X)

    P = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      P[i] = stats.skewnorm.pdf(Y[i], skews[i], loc=locs[i], scale=scales[i])
    return P

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

        Args:
          X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
          Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

        Returns:
         P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
        """
    X, Y = self._handle_input_dimensionality(X, Y)

    locs, scales, skews = self._loc_scale_skew_mapping(X)

    P = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      P[i] = stats.skewnorm.cdf(Y[i], skews[i], loc=locs[i], scale=scales[i])
    return P

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    X = self._handle_input_dimensionality(X)

    locs, scales, skews = self._loc_scale_skew_mapping(X)

    rvs = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      rvs[i] = stats.skewnorm.rvs(skews[i], loc=locs[i], scale=scales[i], random_state=self.random_state)
    rvs = np.expand_dims(rvs, 1)
    assert rvs.shape == (X.shape[0], self.ndim_y)
    return rvs

  def simulate(self, n_samples=1000):
    """ Draws random samples from the unconditional distribution p(x,y)

       Args:
         n_samples: (int) number of samples to be drawn from the conditional distribution

       Returns:
         (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    X = self._sample_x(n_samples)

    assert X.shape == (n_samples, self.ndim_x)
    return X, self.simulate_conditional(X)

  def mean_(self, x_cond, n_samples=None):
    """ Conditional mean of the distribution
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    x = self._handle_input_dimensionality(x_cond)
    locs, _, _ = self._loc_scale_skew_mapping(x)
    assert locs.shape == (x_cond.shape[0], self.ndim_y)
    return locs

def sigmoid(x):
  return 1 / (1+np.exp(-x))

