import itertools
import numpy as np
import scipy.stats as stats

from cde.helpers import sample_center_points, norm_along_axis_1
from .BaseDensityEstimator import BaseDensityEstimator

class LSConditionalDensityEstimation(BaseDensityEstimator):

  def __init__(self, center_sampling_method='k_means', bandwidth=1.0, n_centers=50, regularization=0.1,
               keep_edges=False, random_seed=None):
    """ Least-Squares Density Ratio Estimator

      http://proceedings.mlr.press/v9/sugiyama10a.html

      Args:
          center_sampling_method: String that describes the method to use for finding kernel centers. Allowed values \
                                [all, random, distance, k_means, agglomerative]
          bandwidth: scale / bandwith of the gaussian kernels
          n_centers: Number of kernels to use in the output
          keep_edges: if set to True, the extreme y values as centers are kept (for expressiveness)
          random_seed: (optional) seed (int) of the random number generators used
      """
    self.random_state = np.random.RandomState(seed=random_seed)


    self.center_sampling_method = center_sampling_method
    self.n_centers = n_centers
    self.keep_edges = keep_edges
    self.bandwidth = bandwidth
    self.regularization = regularization

    self.fitted = False
    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = False

  def _build_model(self, X, Y):
    # save variance of data
    self.x_std = np.std(X, axis=0)
    self.y_std = np.std(Y, axis=0)

    # get locations of the gaussian kernel centers
    if self.center_sampling_method == 'all':
      self.n_centers = X.shape[0]

    n_locs = self.n_centers
    X_Y = np.concatenate([X,Y], axis=1)
    centroids = sample_center_points(X_Y, method=self.center_sampling_method, k=n_locs, keep_edges=self.keep_edges)
    self.centr_x = centroids[:,0:self.ndim_x]
    self.centr_y = centroids[:,self.ndim_x:]

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

    # define the full model
    self._build_model(X, Y)

    # determine the kernel weights alpha
    self.h = np.mean(self._gaussian_kernel(X, Y), axis=0)

    a = np.mean(norm_along_axis_1(X,self.centr_x),axis=0)
    b = norm_along_axis_1(self.centr_y, self.centr_y)
    eta = 2 * np.add.outer(a,a) + b

    self.H = (np.sqrt(np.pi)*self.bandwidth) ** self.ndim_y * np.exp( - eta / (5 * self.bandwidth**2))

    self.alpha = np.linalg.solve(self.H + self.regularization * np.identity(self.n_centers), self.h)
    self.alpha[self.alpha < 0] = 0

    self.fitted = True

  def _loss_fun(self, alpha):
    return 0.5 * alpha.T.dot(self.H).dot(alpha) - self.h.T.dot(alpha) + self.regularization * alpha.T.dot(alpha)

  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """
    assert self.fitted, "model must be fitted for predictions"

    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)

    p = np.dot(self.alpha.T, self._gaussian_kernel(X, Y).T)
    p_normalization = (np.sqrt(2*np.pi)*self.bandwidth)**self.ndim_y * np.dot(self.alpha.T, self._gaussian_kernel(X).T)

    return p / p_normalization

  def predict_density(self, X, Y=None, resolution=50):
    """ Computes conditional density p(y|x) over a predefined grid of y target values

      Args:
         X: values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
         Y: (optional) y values to be evaluated from p(y|x) -  if not set, Y will be a grid with with specified resolution
         resulution: integer specifying the resolution of evaluation_runs grid

       Returns: tuple (P, Y)
          - P - density p(y|x) - shape (n_instances, resolution**n_dim_y)
          - Y - grid with with specified resolution - shape (resolution**n_dim_y, n_dim_y) or a copy of Y \
            in case it was provided as argument
    """
    assert X.ndim == 1 or X.shape[1] == self.ndim_x
    X = self._handle_input_dimensionality(X)
    if Y is None:
      y_min = np.min(self.centr_y, axis=0)
      y_max = np.max(self.centr_y, axis=0)
      linspaces = []
      for d in range(self.ndim_y):
        x = np.linspace(y_min[d] - 2.5 * self.bandwidth, y_max[d] + 2.5 * self.bandwidth, num=resolution)
        linspaces.append(x)
      Y = np.asmatrix(list(itertools.product(linspaces[0], linspaces[1])))
    assert Y.ndim == 1 or Y.shape[1] == self.ndim_y

    density = np.zeros(shape=[X.shape[0],Y.shape[0]])
    for i in range(X.shape[0]):
      x = np.tile(X[i,:], (Y.shape[0],1))
      density[i, :] = self.pdf(x, Y)

    return density, Y

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

  def _gaussian_kernel(self, X, Y=None):
    """
    if Y is set returns the product of the gaussian kernels for X and Y, else only the gaussian kernel for X
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

        phi[:, i] = np.exp( - sq_d_x / (2 * self.bandwidth**2)) * np.exp( - sq_d_y / (2 * self.bandwidth**2))
    else:
      for i in range(phi.shape[1]):
        # suqared distances from center point i
        sq_d_x = np.sum(np.square(X - self.centr_x[i, :]), axis=1)
        phi[:, i] = np.exp(- sq_d_x / (2 * self.bandwidth ** 2))

    assert phi.shape == (X.shape[0], self.n_centers)
    return phi

  def _param_grid(self):
    mean_var = np.mean(self.y_std)
    bandwidths = np.asarray([0.01, 0.1, 0.5, 1, 2, 3]) * mean_var

    n_centers = [int(self.n_samples/2), int(self.n_samples/4), int(self.n_samples/10), 50, 20, 10, 5]

    param_grid = {
      "bandwidth": bandwidths,
      "n_centers": n_centers,
      "regularization": [0.01, 0.1],
      "keep_edges": [True, False]
    }
    return param_grid

  def __str__(self):
    return "\nEstimator type: {}\n center sampling method: {}\n n_centers: {}\n keep_edges: {}\n bandwidth: {}\n regularization: {}\n".format(
      self.__class__.__name__, self.center_sampling_method, self.n_centers, self.keep_edges, self.bandwidth, self.regularization)

  def __unicode__(self):
    return self.__str__()