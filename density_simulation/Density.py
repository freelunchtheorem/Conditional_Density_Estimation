
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

  def simulate(self, X):
    """
    Draws random samples from the conditional distribution
    :param X: X to condition on
    :return: random samples
    """