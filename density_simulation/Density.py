import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class ConditionalDensity:

  def pdf(self, X, Y):
    """
    p(y|x)
    :param X: x to be conditioned on
    :param Y: y
    :return: conditional density
    """

    raise NotImplementedError

  def cdf(self, X, Y):
    """
    P(Y < y | x)
    :param X: x to be conditioned on
    :param Y: y
    :return: cumulated conditional density
    """
    raise NotImplementedError


  def simulate_conditional(self, X):
    """
    Draws random samples from the conditional distribution
    :param X: X to be conditioned on
    :return: random samples
    """
    raise NotImplementedError


  def simulate(self, n_samples):
    """
    Draws random samples from the unconditional distribution
    :param n_samples: number of samples to be drawn from the conditional distribution
    :return: random samples
    """
    raise NotImplementedError

  def plot(self, xlim=(-5, 5), ylim=(-5, 5), resolution=100, mode="pdf"):
    """
    Plots the distribution specified in mode if x and y are 1-dimensional each
    :param xlim: 2-tuple specifying the x axis limits
    :param ylim: 2-tuple specifying the y axis limits
    :param resolution: integer specifying the resolution of plot
    :param mode: spefify which dist to plot ["pdf", "cdf", "joint_pdf"]
    :return:
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