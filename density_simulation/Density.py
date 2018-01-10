
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
