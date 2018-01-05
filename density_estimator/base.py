from sklearn.base import BaseEstimator
import numpy as np

class BaseDensityEstimator(BaseEstimator):

  def fit(self, X, Y):
    raise NotImplementedError

  def predict(self, X, Y):
    raise NotImplementedError

  def predict_density(self, X, Y=None, resolution=50):
    raise NotImplementedError

  def score(self, X, Y):
    """
    computes the mean conditional likelihood of the provided data (X, Y)
    :param X: nummpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
    :param Y: nummpy array of y targets - shape: (n_query_samples, n_dim_y)
    :return: mean likelihood
    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    conditional_likelihoods = self.predict(X, Y)
    return np.mean(conditional_likelihoods)