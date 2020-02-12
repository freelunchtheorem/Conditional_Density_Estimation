import itertools
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp

from cde.utils.center_point_select import sample_center_points
from cde.utils.misc import norm_along_axis_1
from .BaseDensityEstimator import BaseDensityEstimator
from cde.utils.async_executor import execute_batch_async_pdf

MULTIPROC_THRESHOLD = 10**4

class LSConditionalDensityEstimation(BaseDensityEstimator):
  """ Least-Squares Density Ratio Estimator

  http://proceedings.mlr.press/v9/sugiyama10a.html

  Args:
      name: (str) name / identifier of estimator
      ndim_x: (int) dimensionality of x variable
      ndim_y: (int) dimensionality of y variable
      center_sampling_method: String that describes the method to use for finding kernel centers. Allowed values \
                            [all, random, distance, k_means, agglomerative]
      bandwidth: scale / bandwith of the gaussian kernels
      n_centers: Number of kernels to use in the output
      regularization: regularization / damping parameter for solving the least-squares problem
      keep_edges: if set to True, the extreme y values as centers are kept (for expressiveness)
      n_jobs: (int) number of jobs to launch for calls with large batch sizes
      random_seed: (optional) seed (int) of the random number generators used
    """

  def __init__(self, name='LSCDE', ndim_x=None, ndim_y=None, center_sampling_method='k_means',
               bandwidth=0.5, n_centers=500, regularization=1.0,
               keep_edges=True, n_jobs=-1, random_seed=None):

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    self.center_sampling_method = center_sampling_method
    self.n_centers = n_centers
    self.keep_edges = keep_edges
    self.bandwidth = bandwidth
    self.regularization = regularization
    self.n_jobs = n_jobs

    self.fitted = False
    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = False

  def _build_model(self, X, Y):
    # save mean and variance of data for normalization
    self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
    self.x_std, self.y_std = np.std(X, axis=0),  np.std(Y, axis=0)

    # get locations of the gaussian kernel centers
    if self.center_sampling_method == 'all':
      self.n_centers = X.shape[0]
    else:
      self.n_centers = min(self.n_centers, X.shape[0])

    n_locs = self.n_centers
    X_Y_normalized = np.concatenate(list(self._normalize(X, Y)), axis=1)
    centroids = sample_center_points(X_Y_normalized, method=self.center_sampling_method, k=n_locs,
                                     keep_edges=self.keep_edges, random_state=self.random_state)
    self.centr_x = centroids[:, 0:self.ndim_x]
    self.centr_y = centroids[:, self.ndim_x:]

    #prepare gaussians for sampling
    self.gaussians_y = [stats.multivariate_normal(mean=center, cov=self.bandwidth) for center in self.centr_y]

    assert self.centr_x.shape == (n_locs, self.ndim_x) and self.centr_y.shape == (n_locs, self.ndim_y)

  def fit(self, X, Y, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
    """
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
    self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]

    self._build_model(X, Y)

    X_normalized, Y_normalized = self._normalize(X, Y)

    # determine the kernel weights alpha
    self.h = np.mean(self._gaussian_kernel(X_normalized, Y_normalized), axis=0)

    a = np.mean(norm_along_axis_1(X_normalized,self.centr_x),axis=0)
    b = norm_along_axis_1(self.centr_y, self.centr_y)
    eta = 2 * np.add.outer(a,a) + b

    self.H = (np.sqrt(np.pi) * self.bandwidth) ** self.ndim_y * np.exp(- eta / (5 * self.bandwidth ** 2))

    self.alpha = np.linalg.solve(self.H + self.regularization * np.identity(self.n_centers), self.h)
    self.alpha[self.alpha <= 0] = 1e-10 # set to small value instead of 0 for numerical stability

    self.fitted = True

  def pdf(self, X, Y):
    """ Predicts the conditional density p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional probability density p(y|x) - numpy array of shape (n_query_samples, )

     """
    assert self.fitted, "model must be fitted for predictions"

    X, Y = self._handle_input_dimensionality(X, Y)

    n_samples = X.shape[0]
    if n_samples >= MULTIPROC_THRESHOLD:
      return execute_batch_async_pdf(self._pdf, X, Y, n_jobs=self.n_jobs)
    else:
      return self._pdf(X, Y)

  def log_pdf(self, X, Y):
    """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

          Args:
            X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
            Y: numpy array of y targets - shape: (n_samples, n_dim_y)

          Returns:
             conditional log-probability density log p(y|x) - numpy array of shape (n_query_samples, )

        """
    assert self.fitted, "model must be fitted for predictions"

    X, Y = self._handle_input_dimensionality(X, Y)

    n_samples = X.shape[0]
    if n_samples >= MULTIPROC_THRESHOLD:
      return execute_batch_async_pdf(self._log_pdf, X, Y, n_jobs=self.n_jobs)
    else:
      return self._log_pdf(X, Y)
    
  def mean_std(self, X, n_samples=10 ** 6):
    """ sample from the conditional mixture distributions - requires the model to be fitted

    Args:
      X: values to be conditioned on  - numpy array of shape (n_instances, n_dim_x)

    Returns: tuple (mean, stddev)
      - mean -  numpy array of shape (n_samples, ndim_y)
      - stddev - - numpy array of shape (n_samples, ndim_y)
    """
    assert self.fitted
    X = self._handle_input_dimensionality(X)

    fact = np.multiply(self.alpha, self._gaussian_kernel(X))
    fact = fact / np.sum(fact, axis=1)[:,None]
    
    mean = fact @ self.centr_y
    print(fact.shape, self.centr_y.shape, mean.shape)
    #return
    expected_variance = self.bandwidth**2
    variance_expectations = (fact @ (self.centr_y**2)) - mean**2
    return (mean, np.sqrt(expected_variance + variance_expectations))
    
  def sample(self, X):
    """ sample from the conditional mixture distributions - requires the model to be fitted

    Args:
      X: values to be conditioned on when sampling - numpy array of shape (n_instances, n_dim_x)

    Returns: tuple (X, Y)
      - X - the values to conditioned on that were provided as argument - numpy array of shape (n_samples, ndim_x)
      - Y - conditional samples from the model p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    assert self.fitted
    X = self._handle_input_dimensionality(X)

    weights = np.multiply(self.alpha, self._gaussian_kernel(X))
    weights = weights / np.sum(weights, axis=1)[:,None]

    Y = np.zeros(shape=(X.shape[0], self.ndim_y))
    for i in range(X.shape[0]):
      discrete_dist = stats.rv_discrete(values=(range(weights.shape[1]), weights[i, :]))
      idx = discrete_dist.rvs()
      Y[i, :] = self.gaussians_y[idx].rvs()

    return X, Y

  def _pdf(self, X, Y):
   return np.exp(self._log_pdf(X, Y))

  def _log_pdf(self, X, Y):
    X_normalized, Y_normalized = self._normalize(X, Y)
    log_p = logsumexp(np.log(self.alpha.T) + self._log_gaussian_kernel(X_normalized, Y_normalized), axis=1)
    log_normalization = (0.5 * np.log(2 *np.pi) + np.log(self.bandwidth)) * self.ndim_y + \
                        logsumexp(np.log(self.alpha.T) + self._log_gaussian_kernel(X_normalized), axis=1)

    return np.squeeze(log_p - log_normalization - np.sum(np.log(self.y_std)))

  def _normalize(self, X, Y):
    X_normalized = (X - self.x_mean) / self.x_std
    Y_normalized = (Y - self.y_mean) / self.y_std
    return X_normalized, Y_normalized

  def _gaussian_kernel(self, X, Y=None):
    """
    if Y is set returns the product of the gaussian kernels for X and Y, else only the gaussian kernel for X
    :param X: numpy array of size (n_samples, ndim_x)
    :param Y: numpy array of size (n_samples, ndim_y)
    :return: phi -  numpy array of size (n_samples, n_centers)
    """
    return np.exp(self._log_gaussian_kernel(X, Y))

  def _log_gaussian_kernel(self, X, Y=None):
    """
    if Y is set returns the sum of the gaussian log-kernels for X and Y, else only the gaussian log-kernel for X
    :param X: numpy array of size (n_samples, ndim_x)
    :param Y: numpy array of size (n_samples, ndim_y)
    :return: phi -  numpy array of size (n_samples, n_centers)
    """
    phi = np.zeros(shape=(X.shape[0], self.n_centers))

    if Y is not None:
      for i in range(phi.shape[1]):

        #suqared distances from center point i
        sq_d_x = np.sum(np.square(X - self.centr_x[i, :]), axis=1)
        sq_d_y = np.sum(np.square(Y - self.centr_y[i, :]), axis=1)

        phi[:, i] = - sq_d_x / (2 * self.bandwidth ** 2) - sq_d_y / (2 * self.bandwidth ** 2)
    else:
      for i in range(phi.shape[1]):
        # suqared distances from center point i
        sq_d_x = np.sum(np.square(X - self.centr_x[i, :]), axis=1)
        phi[:, i] = - sq_d_x / (2 * self.bandwidth ** 2)

    assert phi.shape == (X.shape[0], self.n_centers)
    return phi

  def _param_grid(self):
    param_grid = {
      "n_centers": np.asarray([100, 500, 1000]),
      "bandwidth": np.asarray([0.1, 0.2, 0.5, 0.7, 1.0]),
      "regularization": np.asarray([0.1, 0.5, 1.0, 4.0, 8.0])
    }
    return param_grid

  def __str__(self):
    return "\nEstimator type: {}\n center sampling method: {}\n n_centers: {}\n keep_edges: {}\n bandwidth: {}\n regularization: {}\n ".format(
      self.__class__.__name__, self.center_sampling_method, self.n_centers, self.keep_edges, self.bandwidth, self.regularization)

  def __unicode__(self):
    return self.__str__()
