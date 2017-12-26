import numpy as np
import numpy as np
from sklearn.base import BaseEstimator
from density_estimator.helpers import sample_center_points
import itertools

class NeighborKernelDensityEstimation(BaseEstimator):

  def __init__(self, epsilon=1, bandwidth=1.0, weighted=True):
    self.epsilon = epsilon
    self.weighted = weighted
    self.bandwidth = bandwidth

  def fit(self, X, Y):
    """
    lazy learner - just stores the learning data
    """

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)
    if Y.ndim == 1:
      Y = np.expand_dims(Y, axis=1)

    self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    assert X.ndim == Y.ndim == 2

    self.X_train = X
    self.Y_train = Y

  def predict(self, X, Y):
    self.X_train


