import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from density_estimator.helpers import sample_center_points
import matplotlib.pyplot as plt
from density_simulation.econ_densities import build_econ1_dataset
import itertools
import time

class LQConditionalDensityEstimation(BaseEstimator):

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
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)
    if Y.ndim == 1:
      Y = np.expand_dims(Y, axis=1)

    self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    assert X.ndim == Y.ndim == 2

    # define the full model
    self._build_model(X, Y)

    # calculate weight vector alpha
    # phi_x =compute_kernel_Gaussian(X.flatten(), self.centr_x, sigma=self.bandwidth)
    # phi_y =compute_kernel_Gaussian(Y.flatten(), self.centr_y, sigma=self.bandwidth)
    # phi = np.multiply(phi_x, phi_y)
    #
    # self.H = phi_x.T.dot(phi_y) / Y.shape[0]
    #
    # phi = np.zeros(shape=(X.shape[0], self.n_centers))

    self.h = np.mean(self._gaussian_kernel(X, Y), axis=0)

    a = np.mean(norm_along_axis_1(X,self.centr_x),axis=0)
    b = norm_along_axis_1(self.centr_y, self.centr_y)
    eta = 2 * np.add.outer(a,a) + b

    self.H = (np.sqrt(np.pi)*self.bandwidth) ** self.ndim_y * np.exp( - eta / (5 * self.bandwidth**2))

    self.alpha = np.linalg.solve(self.H + self.regularization * np.identity(self.n_centers), self.h)
    self.alpha[self.alpha < 0] = 0

    loss_fun = 0.5 * self.alpha.T * self.H * self.alpha - self.h.T * self.alpha + \
               self.regularization * self.alpha.T * self.alpha
    # print(loss_fun)
    # print(self.alpha)
    # print(self.centr_x, self.centr_y)

    self.fitted = True

  def predict(self, X, Y):
    """
    copmutes the contitional likelihood p(y|x) given the fitted model
    :param X: nummpy array to be conditioned on
    :param Y:
    :return:
    """
    assert self.fitted, "model must be fitted for predictions"

    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)
    if Y.ndim == 1:
      Y = np.expand_dims(Y, axis=1)

    assert X.ndim == Y.ndim == 2
    assert X.shape[1] == self.ndim_x
    assert Y.shape[1] == self.ndim_y

    print(self.centr_x, self. centr_y)
    print(self.alpha)

    #self._gaussian_kernel(u, v)

    p = np.dot(self.alpha.T, self._gaussian_kernel(X,Y).T)
    p_normalization = (np.sqrt(2*np.pi)*self.bandwidth)**self.ndim_y * np.dot(self.alpha.T, self._gaussian_kernel(X).T)

    return p #/ p_normalization


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
        phi[:, i] = np.exp( - np.linalg.norm(X - self.centr_x[i,:], axis=1, ord=2) / (2 * self.bandwidth**2)) * \
                    np.exp( - np.linalg.norm(Y - self.centr_y[i,:], axis=1, ord=2) / (2 * self.bandwidth**2))
    else:
      for i in range(phi.shape[1]):
        phi[:, i] = np.exp( - np.linalg.norm(X - self.centr_x[i,:], axis=1, ord=2) / (2 * self.bandwidth**2))
    assert phi.shape == (X.shape[0], self.n_centers)
    return phi

def norm_along_axis_1(A,B):
  """
  calculates the euclidean distance along the axis 1 of both 2d arrays
  :param A: numpy array of shape (n, k)
  :param B: numpy array of shape (m, k)
  :return: numpy array of shape (n.m)
  """
  assert A.shape[1] == B.shape[1]
  result = np.zeros(shape=(A.shape[0], B.shape[0]))

  for i in range(B.shape[1]):
    result[:, i] = np.linalg.norm(A - B[i,:], axis=1)

  return result

# TODO: delete the two functions
def compute_kernel_Gaussian(x, centers, sigma):
  result = [[kernel_Gaussian(row, center, sigma) for center in centers] for row in x]
  result = np.matrix(result)
  return result


def kernel_Gaussian(x, y, sigma):
  return np.exp(- np.linalg.norm(x - y) / (2 * sigma * sigma))


# def eucliden_distance_along_axis_1(A, B):
#   """
#   calculates the euclidean distance along the axis 1 of both 2d arrays
#   :param A: numpy array of shape (n, k)
#   :param B: numpy array of shape (m, k)
#   :return: numpy array of shape (n.m)
#   """
#   assert A.shape[1] == B.shape[1]
#
#   d = (A ** 2).sum(axis=-1)[:, np.newaxis] + (B ** 2).sum(axis=-1)
#   d -= 2 * np.squeeze(A.dot(B[..., np.newaxis]), axis=-1)
#   d **= 0.5
#   return d


if __name__ == "__main__":
  n_observations = 2000  # number of data points
  n_features = 3  # number of features

  X_train, X_test, Y_train, Y_test = build_econ1_dataset(n_observations)
  model = LQConditionalDensityEstimation()

  #X_train = np.random.uniform(size=[n_observations, n_features])
  #Y_train = np.random.uniform(size=[n_observations, 2])
  model.fit(X_train, Y_train)

  n_samples = 2000
  X_plot = np.expand_dims(np.zeros(n_samples), axis=1)
  Y_plot = np.linspace(-2, 4, num=n_samples)
  print(X_plot.shape, Y_plot.shape)
  result = model.predict(X_plot, Y_plot)
  print(result.shape)
  plt.plot(Y_plot, result)
  plt.show()



