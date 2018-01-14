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

