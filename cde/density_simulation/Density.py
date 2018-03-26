import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.base import BaseEstimator

class ConditionalDensity(BaseEstimator):

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

  def __str__(self):
    raise NotImplementedError

  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    if np.size(X) == 1 and Y is None:
      return X

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)

    if Y is not None:
      if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

      assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
      assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    if fitting: # store n_dim of training data
      self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    else:
      assert X.shape[1] == self.ndim_x, "X must have shape (?, %i) but provided X has shape %s" % (self.ndim_x, X.shape)
      if Y is not None:
        assert Y.shape[1] == self.ndim_y, "Y must have shape (?, %i) but provided Y has shape %s" % (self.ndim_y, Y.shape)

    if Y is None:
      return X
    else:
      return X, Y

  def get_params(self, deep=True):
    param_dict = super(ConditionalDensity, self).get_params(deep=deep)
    param_dict['simulator'] = self.__class__.__name__
    return param_dict