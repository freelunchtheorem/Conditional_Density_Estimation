import numpy as np
from sklearn.base import BaseEstimator
from density_estimator.helpers import sample_center_points, norm_along_axis_1, handle_input_dimensionality
import itertools

class LSConditionalDensityEstimation(BaseEstimator):

  def __init__(self, center_sampling_method='k_means', bandwidth=1.0, n_centers=50, regularization=0.1, keep_edges=False):
    """
    Main class for the Least Squares Conditional Density Estimation
    :param center_sampling_method:
    :param n_centers:
    :param keep_edges: Keep the extreme y values as center to keep expressiveness
    """
    self.center_sampling_method = center_sampling_method
    self.n_centers = n_centers
    self.keep_edges = keep_edges
    self.bandwidth = bandwidth
    self.regularization = regularization

    self.fitted = False

  def _build_model(self, X, Y):
    # get locations of the gaussian kernel centers
    n_locs = self.n_centers
    X_Y = np.concatenate([X,Y], axis=1)
    centroids = sample_center_points(X_Y, method=self.center_sampling_method, k=n_locs, keep_edges=self.keep_edges)
    self.centr_x = centroids[:,0:self.ndim_x]
    self.centr_y = centroids[:,self.ndim_x:]
    assert self.centr_x.shape == (n_locs, self.ndim_x) and self.centr_y.shape == (n_locs, self.ndim_y)

  def fit(self, X, Y, **kwargs):
    """
    fits the model by determining the weight vector alpha
    :param X: nummpy array to be conditioned on - shape: (n_samples, n_dim_x)
    :param Y: nummpy array of y targets - shape: (n_samples, n_dim_y)
    """
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    X, Y = handle_input_dimensionality(X, Y)
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

  def predict(self, X, Y):
    """
    copmutes the contitional likelihood p(y|x) given the fitted model
    :param X: nummpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
    :param Y: nummpy array of y targets - shape: (n_query_samples, n_dim_y)
    :return: numpy array of shape (n_query_samples, ) holding the conditional likelihood p(y|x)
    """
    assert self.fitted, "model must be fitted for predictions"

    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    X, Y = handle_input_dimensionality(X, Y)
    assert X.shape[1] == self.ndim_x
    assert Y.shape[1] == self.ndim_y

    p = np.dot(self.alpha.T, self._gaussian_kernel(X, Y).T)
    p_normalization = (np.sqrt(2*np.pi)*self.bandwidth)**self.ndim_y * np.dot(self.alpha.T, self._gaussian_kernel(X).T)

    return p / p_normalization

  def predict_density(self, X, Y=None, resolution=50):
    """
    conditional density p(y|x) over a predefined grid of target values
    :param X values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
    :param (optional) Y - y values to be evaluated from p(y|x) -  if not set, Y will be a grid with with specified resolution
    :param resulution of evaluation grid
    :return density p(y|x) shape: (n_instances, resolution**n_dim_y), Y - grid with with specified resolution - shape: (resolution**n_dim_y, n_dim_y)
    """
    assert X.shape[1] == self.ndim_x

    if Y is None:
      y_min = np.min(self.centr_y, axis=0)
      y_max = np.max(self.centr_y, axis=0)
      linspaces = []
      for d in range(self.ndim_y):
        x = np.linspace(y_min[d] - 2.5 * self.bandwidth, y_max[d] + 2.5 * self.bandwidth, num=resolution)
        linspaces.append(x)
      Y = np.asmatrix(list(itertools.product(linspaces[0], linspaces[1])))

    assert Y.shape[1] == self.ndim_y

    density = np.zeros(shape=[X.shape[0],Y.shape[0]])
    for i in range(X.shape[0]):
      x = np.tile(X[i,:], (Y.shape[0],1))
      density[i, :] = self.predict(x, Y)

    return density, Y

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



