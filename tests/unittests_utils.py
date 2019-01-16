import unittest
from scipy.stats import norm
import warnings
import pickle
import tensorflow as tf
import sys
import os
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.helpers import sample_center_points, mc_integration_cauchy, norm_along_axis_1
from cde.utils.async_executor import execute_batch_async_pdf


class TestHelpers(unittest.TestCase):

  """ sample center points """

  def test_1_shape_center_point(self):
    methods = ["all", "random", "k_means" , "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(120,2))
      centers = sample_center_points(Y, method=m, k=50)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  def test_1_shape_center_point_k_means(self):
    Y = np.asarray([1.0, 2.0])
    centers = sample_center_points(Y, method="k_means", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_1_shape_center_point_agglomerative(self):
    Y = np.random.uniform(size=[20,3])
    centers = sample_center_points(Y, method="agglomerative", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_1_shape_center_point_keep_edges(self):
    methods = ["random", "k_means", "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(100,2))
      centers = sample_center_points(Y, method=m, k=5, keep_edges=True)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  """ norm along axis """

  def test_2_norm_along_axis_1(self):
    A = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    B = np.asarray([[0.0,0.0], [0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_2_norm_along_axis_2(self):
    A = np.asarray([[1.0, 0.0]])
    B = np.asarray([[0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_2_norm_along_axis_3(self):
    A = np.random.uniform(size=[20, 3])
    B = np.random.uniform(size=[10, 3])
    dist = norm_along_axis_1(A, B, squared=True)
    self.assertEqual(dist.shape, (20,10))

  """ monte carlo integration """

  def test_mc_integration_chauchy_1(self):
    func = lambda y: np.expand_dims(stats.multivariate_normal.pdf(y, mean=[0, 0], cov=np.diag([2, 2])), axis=1)
    integral = mc_integration_cauchy(func, ndim=2, n_samples=10 ** 7, batch_size=10**6)
    self.assertAlmostEqual(1.0, integral[0], places=2)

  def test_mc_integration_chauchy_2(self):
    func = lambda y: y * np.tile(np.expand_dims(stats.multivariate_normal.pdf(y, mean=[1, 2], cov=np.diag([2, 2])), axis=1), (1,2))
    integral = mc_integration_cauchy(func, ndim=2, n_samples=10 ** 7, batch_size=10**6)
    self.assertAlmostEqual(1, integral[0], places=2)
    self.assertAlmostEqual(2, integral[1], places=2)

class TestExecAsyncBatch(unittest.TestCase):

  def test_batch_exec_1(self):
    def pdf(X, Y):
      return Y[:, 0]

    n_queries = 10**3
    X = np.ones((n_queries, 2)) * 2
    Y = np.stack([np.linspace(-3, 3, num=n_queries), np.linspace(-3, 3, num=n_queries)], axis=-1)
    p_true = pdf(X, Y)

    p_batched = execute_batch_async_pdf(pdf, X, Y, batch_size=10000)

    self.assertLessEqual(np.mean((p_true - p_batched)**2), 0.00001)

  def test_batch_exec_2(self):
    from scipy.stats import multivariate_normal

    def pdf(X, Y):
      std = 1
      ndim_y = Y.shape[1]
      return multivariate_normal.pdf(Y, mean=np.zeros(ndim_y), cov=np.eye(ndim_y)*std**2)

    n_queries = 8*10 ** 4
    X = np.ones((n_queries, 2)) * 2
    Y = np.stack([np.linspace(-3, 3, num=n_queries), np.linspace(-3, 3, num=n_queries)], axis=-1)
    p_true = pdf(X, Y)

    p_batched = execute_batch_async_pdf(pdf, X, Y, batch_size=10000, n_jobs=8)

    self.assertLessEqual(np.mean((p_true - p_batched) ** 2), 0.00001)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(TestHelpers())
  suite.addTest(TestExecAsyncBatch())
  return suite

if __name__ == '__main__':
  warnings.filterwarnings("ignore")

  testmodules = [
   'unittests_estimators.TestHelpers',
   'unittests_estimators.TestExecAsyncBatch',
   ]
  suite = unittest.TestSuite()
  for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

  unittest.TextTestRunner().run(suite)
