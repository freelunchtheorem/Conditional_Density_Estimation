import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import scipy.optimize as optimize
import warnings

from cde.utils.misc import norm_along_axis_1
from .BaseDensityEstimator import BaseDensityEstimator
from cde.utils.async_executor import execute_batch_async_pdf
from scipy.special import logsumexp

_MULTIPROC_THRESHOLD = 10 ** 4
_N_POINT_OUT_OF_RANGE = 5 # number for closest points to consider if all points are outside of the epsilon range
_DAFAULT_EPSILON = 0.4
_MAX_ITER_CV_ML_OPTIMIZER = 30

class NeighborKernelDensityEstimation(BaseDensityEstimator):
  """
  Epsilon-Neighbor Kernel Density Estimation (lazy learner) with Gaussian Kernels

  Args:
    name: (str) name / identifier of estimator
    ndim_x: (int) dimensionality of x variable
    ndim_y: (int) dimensionality of y variable
    epsilon: size of the (normalized) neighborhood region
    bandwidth: (float of array_like) initial bandwidth parameter
    param_selection: parameter selection method. Must be
                     - None or False: use the provided epsilon and bandwidth
                     - normal_reference: bandwidths are chosen according to normal reference distribution
                     - cv_ml: select bandwidth and epsilon via maximum likelihood leave-one-out cross-validation
    weighted: if true - the neighborhood Gaussians are weighted according to their distance to the query point,
              if false - all neighborhood Gaussians are weighted equally
    random_seed: (optional) seed (int) of the random number generators used

  """

  def __init__(self, name='NKDE', ndim_x=None, ndim_y=None, epsilon=0.4, bandwidth=0.6, param_selection='normal_reference',
               weighted=True, n_jobs=-1, random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)

    assert isinstance(bandwidth, (int, float)) or isinstance(bandwidth, np.ndarray)
    assert isinstance(epsilon, (int, float))
    assert param_selection in ['cv_ml', 'normal_reference', None, False]

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y

    self.random_seed = random_seed

    self.epsilon = epsilon
    self.bandwidth = bandwidth
    self.param_selection = param_selection
    self.weighted = weighted
    self.n_jobs = n_jobs

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

  def pdf(self, X, Y):
    """ Predicts the conditional probability density p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional probability p(y|x) - numpy array of shape (n_query_samples, )

    """
    return np.exp(self.log_pdf(X,Y))

  def log_pdf(self, X, Y):
    """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional log-probability log p(y|x) - numpy array of shape (n_query_samples, )

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    n_samples = X.shape[0]
    if n_samples >= _MULTIPROC_THRESHOLD:
      return execute_batch_async_pdf(self._log_pdf, X, Y, n_jobs=self.n_jobs)
    else:
      return self._log_pdf(X, Y)

  def sample(self, X):
    raise NotImplementedError("Neighbor Kernel Density Estimation is a lazy learner and does not support sampling")

  def loo_likelihood(self, bandwidth, epsilon):
    """
    calculates the negative leave-one-out log-likelihood of the training data

    Args:
      bw: bandwidth parameter
      epsilon: size of the (normalized) neighborhood region
    """
    kernel_weights = self._kernel_weights(self.X_train, epsilon)

    # remove kernel of query x and re-normalize weights
    np.fill_diagonal(kernel_weights, 0.0)
    kernel_weights_loo = kernel_weights / np.sum(kernel_weights, axis=-1, keepdims=True)

    conditional_log_densities = np.zeros(self.n_train_points)
    for i in range(self.n_train_points):
      conditional_log_densities[i] = self._log_density(bandwidth, kernel_weights_loo[i, :], self.Y_train[i, :])

    return np.mean(conditional_log_densities)

  def _build_model(self, X, Y):
    # save mean and std of data for normalization
    self.x_std = np.std(X, axis=0)
    self.x_mean = np.mean(X, axis=0)
    self.y_mean = np.std(Y, axis=0)
    self.y_std = np.std(Y, axis=0)

    self.n_train_points = X.shape[0]

    # lazy learner - just store training data
    self.X_train = self._normalize_x(X)
    self.Y_train = Y

    # prepare Gaussians centered in the Y points
    self.locs_array = np.vsplit(Y, self.n_train_points)
    self.log_kernel = multivariate_normal(mean=np.ones(self.ndim_y)).logpdf

    # select / properly initialize bandwidth and epsilon
    if isinstance(self.bandwidth, (int, float)):
      self.bandwidth = self.y_std * self.bandwidth

    if self.param_selection == 'normal_reference':
      self.bandwidth = self._normal_reference()
    elif self.param_selection == 'cv_ml':
      self.bandwidth, self.epsilon = self._cv_ml()

  def _log_pdf(self, X, Y):
    """ 1. Determine weights of the Gaussians """
    X_normalized = self._normalize_x(X)
    kernel_weights = self._kernel_weights(X_normalized, self.epsilon)

    """ 2. Calculate the conditional log densities """
    n_samples = X.shape[0]

    conditional_densities = np.zeros(n_samples)
    for i in range(n_samples):
      conditional_densities[i] = self._log_density(self.bandwidth, kernel_weights[i, :], Y[i, :])

    return conditional_densities

  def _kernel_weights(self, X_normalized, epsilon):
    X_dist = norm_along_axis_1(X_normalized, self.X_train, norm_dim=True)
    mask = X_dist > epsilon
    num_neighbors = np.sum(np.logical_not(mask), axis=1)

    # Extra treatment for X that are outside of the epsilon range
    if np.any(num_neighbors <= _N_POINT_OUT_OF_RANGE):
      for i in np.nditer(np.where(num_neighbors <= _N_POINT_OUT_OF_RANGE)): # if all points outside of epsilon region - take closest points
        closest_indices = np.argsort(X_dist[i, :])[:_N_POINT_OUT_OF_RANGE]
        for j in np.nditer(closest_indices):
          mask[i,j] = False

    num_neighbors = np.sum(np.logical_not(mask), axis=1)
    neighbor_distances = np.ma.masked_where(mask, X_dist)

    if self.weighted:
      # neighbors are weighted in proportion to their distance to the query point
      neighbor_weights = normalize(neighbor_distances.filled(fill_value=0), norm='l1', axis=1)
    else:
      # all neighbors are weighted equally
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # don't print division by zero warning
        weights = 1 / num_neighbors
      neighbor_weights = neighbor_distances.copy()
      neighbor_weights[:] = weights[:, None]
      neighbor_weights = np.ma.masked_where(mask, neighbor_weights).filled(fill_value=0)

    return neighbor_weights

  def _log_density(self, bw, neighbor_weights, y):
    assert neighbor_weights.shape[0] == self.n_train_points
    assert y.shape[0] == self.ndim_y

    log_single_densities = np.array([self._single_log_density(bw, neighbor_weights[i], i, y)
                                 for i in np.nditer(np.nonzero(neighbor_weights))])

    return logsumexp(log_single_densities)

  def _single_log_density(self, bw, neighbor_weight, kernel_id, y):
      return np.log(neighbor_weight) + self.log_kernel((y - self.Y_train[kernel_id, :]) / bw) - np.sum(np.log(bw))

  def _param_grid(self):
    mean_std_y = np.mean(self.y_std)
    mean_std_x = np.mean(self.x_std)
    bandwidths = np.asarray([0.2, 0.5, 0.7]) * mean_std_y
    epsilons = np.asarray([0.05, 0.2, 0.4]) * mean_std_x

    param_grid = {
      "bandwidth": bandwidths,
      "epsilon": epsilons,
      "weighted": [True, False]
    }
    return param_grid

  def _normalize_x(self, X):
    X_normalized = (X - self.x_mean) / self.x_std
    assert X_normalized.shape == X.shape
    return X_normalized

  def _normal_reference(self):
    X_dist = norm_along_axis_1(self.X_train, self.X_train, norm_dim=True)

    # filter out all points that are not in a epsilon region of x
    avg_num_neighbors = np.mean(np.ma.masked_where(X_dist > self.epsilon, X_dist).count(axis=1)) - 1

    return 1.06 * self.y_std * avg_num_neighbors ** (- 1. / (4 + self.ndim_y))

  def _cv_ml(self):
    bw = self._normal_reference() # use normal_reference as initialization for bandwidth
    x0 = np.concatenate([bw, np.array([self.epsilon])], axis=0)

    def loologli(x):
      assert x.shape[0] == self.ndim_y + 1
      bw = x[:self.ndim_y]
      eps = float(x[self.ndim_y])
      return - self.loo_likelihood(bw, epsilon=eps)

    x_opt = optimize.fmin(loologli, x0=x0, maxiter=_MAX_ITER_CV_ML_OPTIMIZER, disp=0)
    bw_opt, eps_opt = x_opt[:self.ndim_y], x_opt[self.ndim_y]
    return bw_opt, eps_opt

  def __str__(self):
    return "\nEstimator type: {}\n  epsilon: {}\n weighted: {}\n bandwidth: {}\n ".format(self.__class__.__name__, self.epsilon, self.weighted,
                                                                                         self.bandwidth)

  def __unicode__(self):
    return self.__str__()