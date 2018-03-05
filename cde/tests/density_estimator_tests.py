import unittest
import numpy as np
from scipy.stats import norm
import warnings

from cde.density_estimator.helpers import *
from cde.density_estimator import KernelMixtureNetwork, NeighborKernelDensityEstimation, LSConditionalDensityEstimation

class TestHelpers(unittest.TestCase):

  """ sample center points """

  def test_shape_center_point(self):
    methods = ["all", "random", "k_means" , "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(120,2))
      centers = sample_center_points(Y, method=m, k=50)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  def test_shape_center_point_k_means(self):
    Y = np.asarray([1.0, 2.0])
    centers = sample_center_points(Y, method="k_means", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_shape_center_point_agglomerative(self):
    Y = np.random.uniform(size=[20,3])
    centers = sample_center_points(Y, method="agglomerative", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_shape_center_point_keep_edges(self):
    methods = ["random", "k_means", "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(100,2))
      centers = sample_center_points(Y, method=m, k=5, keep_edges=True)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  """ norm along axis """

  def test_1_norm_along_axis_1(self):
    A = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    B = np.asarray([[0.0,0.0], [0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_1_norm_along_axis_2(self):
    A = np.asarray([[1.0, 0.0]])
    B = np.asarray([[0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_1_norm_along_axis_3(self):
    A = np.random.uniform(size=[20, 3])
    B = np.random.uniform(size=[10, 3])
    dist = norm_along_axis_1(A, B, squared=True)
    self.assertEqual(dist.shape, (20,10))


  """ Conditional Density Estimators """

  def test_NKDE_with_2d_gaussian(self):
    np.random.seed(22)
    data = np.random.normal([2, 2], 1, size=(2000, 2))
    X = data[:, 1]
    Y = data[:, 1]

    model = NeighborKernelDensityEstimation(epsilon=0.1)
    model.fit(X, Y)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.predict(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.01)


  def test_LSCD_with_2d_gaussian(self):
    np.random.seed(22)
    data = np.random.normal([2, 2], 1, size=(2000, 2))
    X = data[:, 1]
    Y = data[:, 1]

    for method in ["all", "k_means"]:
      model = LSConditionalDensityEstimation(center_sampling_method=method, n_centers=400, bandwidth=0.5)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.predict(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


  def test_KMN_with_2d_gaussian(self):
    np.random.seed(22)
    data = np.random.normal([2, 2], 1, size=(2000, 2))
    X = data[:, 1]
    Y = data[:, 1]

    for method in ["agglomerative"]:
      model = KernelMixtureNetwork(center_sampling_method=method, n_centers=100)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.predict(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)






if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  unittest.main()