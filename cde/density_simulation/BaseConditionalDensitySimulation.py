import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.base import BaseEstimator
from cde import ConditionalDensity

class BaseConditionalDensitySimulation(ConditionalDensity):

  def pdf(self, X, Y):
    """ Conditional probability density function p(y|x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the pdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
      p(X|Y) conditional density values for the provided X and Y - numpy array of shape (n_points, )
    """

    raise NotImplementedError

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
     P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
    """

    raise NotImplementedError

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    raise NotImplementedError

  def simulate(self, n_samples):
    """ Draws random samples from the unconditional distribution p(x,y)

    Args:
      n_samples: (int) number of samples to be drawn from the conditional distribution

    Returns:
      (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    raise NotImplementedError

  def plot(self, xlim=(-5, 5), ylim=(-5, 5), resolution=100, mode="pdf"):
    """ Plots the distribution specified in mode if x and y are 1-dimensional each

    Args:
      xlim: 2-tuple specifying the x axis limits
      ylim: 2-tuple specifying the y axis limits
      resolution: integer specifying the resolution of plot
      mode: spefify which dist to plot ["pdf", "cdf", "joint_pdf"]

    """
    modes = ["pdf", "cdf", "joint_pdf"]
    assert mode in modes, "mode must be on of the following: " + modes
    assert self.ndim == 2, "Can only plot two dimensional distributions"

    # prepare mesh
    linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
    linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()

    # calculate values of distribution
    if mode == "pdf":
      Z = self.pdf(X, Y)
    elif mode == "cdf":
      Z = self.cdf(X, Y)
    elif mode == "joint_pdf":
      Z = self.joint_pdf(X, Y)

    X, Y, Z = X.reshape([resolution, resolution]), Y.reshape([resolution, resolution]), Z.reshape(
      [resolution, resolution])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=resolution, ccount=resolution,
                           linewidth=100, antialiased=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

  def mean_(self, x_cond, n_samples=10**7):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert x_cond.ndim == 2

    if self.can_sample:
      return self._mean_mc(x_cond, n_samples=n_samples)
    else:
      return self._mean_pdf(x_cond)

  def covariance(self, x_cond, n_samples=10**7):
    """ Covariance of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      n_samples: number of samples for monte carlo evaluation

    Returns:
      Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    if self.has_pdf:
      return self._covariance_pdf(x_cond)
    elif self.can_sample:
      return self._covariance_mc(x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
    """ Computes the Value-at-Risk (VaR) of the fitted distribution. Only if ndim_y = 1

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      alpha: quantile percentage of the distribution
      n_samples: number of samples for monte carlo evaluation

    Returns:
       VaR values for each x to condition on - numpy array of shape (n_values)
    """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    if self.has_cdf:
      return self._quantile_cdf(x_cond, alpha=alpha)
    elif self.can_sample:
      return self._quantile_mc(x_cond, alpha=alpha, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of the fitted distribution. Only if ndim_y = 1

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
         alpha: quantile percentage of the distribution
         n_samples: number of samples for monte carlo evaluation

       Returns:
         CVaR values for each x to condition on - numpy array of shape (n_values)
       """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)

    if self.has_pdf:
      return self._conditional_value_at_risk_mc_pdf(VaRs, x_cond, alpha=alpha, n_samples=n_samples)
    elif self.can_sample:
      return self._conditional_value_at_risk_sampling(VaRs, x_cond, alpha=alpha, n_samples=n_samples)
    else:
      raise NotImplementedError("Distribution object must either support pdf or sampling in order to compute CVaR")

  def get_params(self, deep=True):
    param_dict = super(BaseConditionalDensitySimulation, self).get_params(deep=deep)
    param_dict['simulator'] = self.__class__.__name__
    return param_dict

