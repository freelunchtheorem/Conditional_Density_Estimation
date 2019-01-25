import scipy.stats as stats
import numpy as np
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation
from cde.utils.distribution import batched_univ_t_cdf, batched_univ_t_pdf, batched_univ_t_rvs

class LinearStudentT(BaseConditionalDensitySimulation):
  """
  A conditional student-t distribution where

  x ~ N(x | 0,I)
  y ~ Student-t(y | dof(x), loc(x), scale(x))
  with
  dof(x) = dof_low + (dof_high - dof_low) * sigmoid(- 2*x)
  loc(x) = mu_slope*x+mu
  scale(x) = std_slope*x+std

  Args:
    ndim_x: number of dimensions of x
    mu: the intercept of the mean line
    mu_slope: the slope of the mean line
    std: the intercept of the std dev. line
    std_slope: the slope of the std. dev. line
    random_seed: seed for the random_number generator
  """

  def __init__(self, ndim_x=10, mu=0.0, mu_slope=0.005, std=0.01, std_slope=0.002, dof_low=2, dof_high=10, random_seed=None):
    assert std > 0
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    self.mu = mu
    self.std = std
    self.mu_slope = mu_slope
    self.std_slope = std_slope
    self.dof_low = dof_low
    self.dof_high = dof_high
    self.ndim_x = ndim_x
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
    loc, scale, dof = self._loc_scale_dof_mapping(X)
    p = batched_univ_t_pdf(Y, loc, scale, dof)
    assert p.shape == (X.shape[0],)
    return p

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

       Args:
         X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
         Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

       Returns:
        P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
    """
    X, Y = self._handle_input_dimensionality(X, Y)
    loc, scale, dof = self._loc_scale_dof_mapping(X)
    p = batched_univ_t_cdf(Y, loc, scale, dof)
    assert p.shape == (X.shape[0],)
    return p

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    X = self._handle_input_dimensionality(X)
    loc, scale, dof = self._loc_scale_dof_mapping(X)
    Y = batched_univ_t_rvs(loc, scale, dof, random_state=self.random_state)
    Y = Y.reshape((-1, 1))
    return X, Y

  def simulate(self, n_samples=1000):
    """ Draws random samples from the joint distribution p(x,y)
    Args:
      n_samples: (int) number of samples to be drawn from the joint distribution

    Returns:
      (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    assert n_samples > 0
    X = self.random_state.normal(loc=0, scale=1, size=(n_samples, self.ndim_x))
    return self.simulate_conditional(X)

  def mean_(self, x_cond, n_samples=None):
    """ Conditional mean of the distribution
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x
    x_cond = self._handle_input_dimensionality(x_cond)
    return self._loc(x_cond)

  def std_(self, x_cond, n_samples=None):
    """ Standard deviation of the distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Conditional standard deviations Std[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x
    x_cond = self._handle_input_dimensionality(x_cond)
    loc, scale, dof = self._loc_scale_dof_mapping(x_cond)
    std = scale * np.sqrt(dof / (dof - 2)).reshape((x_cond.shape[0], self.ndim_y))
    return std

  def _loc_scale_dof_mapping(self, X):
    return self._loc(X), self._scale(X), self._dof(X)

  def _loc(self, X):
    return np.expand_dims(self.mu + np.mean(self.mu_slope * X, axis=-1), axis=-1)

  def _scale(self, X):
    return np.expand_dims(self.std + np.mean(self.std_slope * np.abs(X), axis=-1), axis=-1)

  def _dof(self, X):
    return self.dof_low + (self.dof_high - self.dof_low) * _sigmoid(- 2 * np.mean(X, axis=-1))

  def __str__(self):
    return "\nProbabilistic model type: {}\n std: {}\n n_dim_x: {}\n n_dim_y: {}\n".format(self.__class__.__name__, self.std, self.ndim_x,
                                                                                                self.ndim_y)

  def __unicode__(self):
    return self.__str__()

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))