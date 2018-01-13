import numpy as np

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

  def plot(self, xlim, ylim):
    """
    Plots the density function
    :param xlim: 2-tuple with the x-axis limits
    :param ylim: 2-tuple with the y-axis limits
    """
    raise NotImplementedError

  def _handle_input_dimensionality(self, X, Y, fitting=False):
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)
    if Y.ndim == 1:
      Y = np.expand_dims(Y, axis=1)

    assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
    assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    assert X.shape[1] == self.ndim_x, "X must have shape (?, %i) but provided X has shape %s" % (self.ndim_x, X.shape)
    assert Y.shape[1] == self.ndim_y, "Y must have shape (?, %i) but provided Y has shape %s" % (self.ndim_y, Y.shape)

    return X, Y
