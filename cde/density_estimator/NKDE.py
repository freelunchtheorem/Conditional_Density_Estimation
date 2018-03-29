import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import warnings

from cde.helpers import norm_along_axis_1
from .BaseDensityEstimator import BaseDensityEstimator



class NeighborKernelDensityEstimation(BaseDensityEstimator):
  """
  Epsilon-Neighbor Kernel Density Estimation (lazy learner) with Gaussian Kernels

  Args:
    epsilon: size of the neighborhood region
    bandwidth: scale of the Gaussians
    weighted: if true - the neighborhood Gaussians are weighted according to their distance to the query point,
              if false - all neighborhood Gaussians are weighted equally
    random_seed: (optional) seed (int) of the random number generators used

  """

  def __init__(self, epsilon=5.0, bandwidth=1.0, weighted=True, random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)

    self.epsilon = epsilon
    self.weighted = weighted
    self.bandwidth = bandwidth

    self.fitted = False

    self.can_sample = False
    self.has_pdf = True
    self.has_cdf = False

  def fit(self, X, Y, **kwargs):
    """ Since NKDE is a lazy learner, fit just stores the provided training data (X,Y)

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    self._build_model(X, Y)

    self.can_sample = False
    self.has_cdf = False

    self.fitted = True


  def _build_model(self, X, Y):
    # save variance of data
    self.x_std = np.std(X, axis=0)
    self.y_std = np.std(Y, axis=0)

    self.n_train_points = X.shape[0]

    # lazy learner - just store training data
    self.X_train = X
    self.Y_train = Y

    # prepare Gaussians centered in the Y points
    self.locs_array = np.vsplit(Y, self.n_train_points)
    self.scales = np.diag(self.bandwidth * np.ones(self.ndim_y))
    self.components = [multivariate_normal(mean=loc, cov=self.scales) for loc in self.locs_array]


  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """

    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
    assert X.shape[1] == self.ndim_x
    assert Y.shape[1] == self.ndim_y

    """ 1. Determine weights of the Gaussians """
    X_dist = norm_along_axis_1(X, self.X_train)

    # filter out all points that are not in a epsilon region of x
    mask = X_dist > self.epsilon
    neighbor_distances = np.ma.masked_where(mask, X_dist)
    num_neighbors = neighbor_distances.count(axis=1)

    if self.weighted:
      # neighbors are weighted in proportion to their distance to the query point
      neighbor_weights = normalize(neighbor_distances.filled(fill_value=0), norm='l1', axis=1)
    else:
      # all neighbors are weighted equally
      with warnings.catch_warnings():
        warnings.simplefilter("ignore") # don't print division by zero warning
        weights = 1 / num_neighbors
      neighbor_weights = neighbor_distances.copy()
      neighbor_weights[:] = weights[:, None]
      neighbor_weights = np.ma.masked_where(mask, neighbor_weights)

      neighbor_weights = neighbor_weights.filled(fill_value=0)

    """ 2. Calculate the conditional densities """
    n_samples = X.shape[0]

    conditional_densities = np.zeros(n_samples)
    for i in range(n_samples):
      conditional_densities[i] = self._density(neighbor_weights[i, :], Y[i, :])

    return conditional_densities

  def sample(self, X):
    raise NotImplementedError("Neighbor Kernel Density Estimation is a lazy learner and does not support sampling")

  def _density(self, neighbor_weights, y):
    assert neighbor_weights.shape[0] == self.n_train_points
    assert y.shape[0] == self.ndim_y
    kernel_ids = np.arange(self.n_train_points)

    # vectorized function
    single_densities = np.vectorize(self._single_density, otypes=[np.float])

    # call vectorized function
    single_den = single_densities(neighbor_weights, kernel_ids, y)

    return np.sum(single_den)


  def _single_density(self, neighbor_weight, kernel_id, y):
    if neighbor_weight > 0:
      return neighbor_weight * self.components[kernel_id].pdf(y)
    else:
      return 0

  def _param_grid(self):
    mean_std_y = np.mean(self.y_std)
    mean_std_x = np.mean(self.x_std)
    bandwidths = np.asarray([0.01, 0.1, 0.5, 1, 2, 5]) * mean_std_y
    epsilons = np.asarray([0.001, 0.1, 0.5, 1]) * mean_std_x

    param_grid = {
      "bandwidth": bandwidths,
      "epsilon": epsilons,
      "weighted": [True, False]
    }
    return param_grid


  def __str__(self):
    return "\nEstimator type: {}\n  epsilon: {}\n weighted: {}\n bandwidth: {}\n".format(self.__class__.__name__, self.epsilon, self.weighted,
                                                                                             self.bandwidth)

  def __unicode__(self):
    return self.__str__()