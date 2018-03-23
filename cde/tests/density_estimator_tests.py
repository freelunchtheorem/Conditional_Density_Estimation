import unittest
from scipy.stats import norm
import warnings

from cde.density_estimator.helpers import *
from cde.density_estimator import *
from cde.tests.Dummies import *

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


class TestRiskMeasures(unittest.TestCase):
  def test_value_at_risk_mc(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.01
    VaR_est = est.value_at_risk(x_cond=np.array([0,1]), alpha=alpha)
    VaR_true = norm.ppf(alpha, loc=0, scale=1)
    self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
    self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_value_at_risk_cdf(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=True)

    alpha = 0.05
    VaR_est = est.value_at_risk(x_cond=np.array([0, 1]), alpha=alpha)
    VaR_true = norm.ppf(alpha, loc=0, scale=1)
    self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
    self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_conditional_value_at_risk_mc(self):
    # prepare estimator dummy
    mu = 0
    sigma = 1
    mu1 = np.array([mu])
    sigma1 = np.identity(n=1) * sigma
    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.02

    CVaR_true = mu - sigma/alpha * norm.pdf(norm.ppf(alpha, loc=0, scale=1))
    CVaR_est = est.conditional_value_at_risk(x_cond=np.array([0, 1]), alpha=alpha)

    self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
    self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_mean_mc(self):
    # prepare estimator dummy
    mu = np.array([0,1])
    sigma = np.identity(n=2) * 1
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, has_cdf=False)

    mean_est = est.mean_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)

  def test_mean_mixture(self):
    np.random.seed(22)
    data = np.random.normal([2, 2, 7, -2], 1, size=(5000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork(n_centers=5)
    model.fit(X, Y)

    mean_est = model.mean_(x_cond=np.array([[1,2]]))
    self.assertAlmostEqual(mean_est[0][0], 7, places=1)
    self.assertAlmostEqual(mean_est[0][1], -2, places=1)


class TestConditionalDensityEstimators_2d_gaussian(unittest.TestCase):

  def get_samples(self, std=1.0):
    np.random.seed(22)
    data = np.random.normal([2, 2], std, size=(2000, 2))
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_NKDE_with_2d_gaussian(self):
    X, Y = self.get_samples()

    model = NeighborKernelDensityEstimation(epsilon=0.1)
    model.fit(X, Y)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_LSCD_with_2d_gaussian(self):
    X, Y = self.get_samples()

    for method in ["all", "k_means"]:
      model = LSConditionalDensityEstimation(center_sampling_method=method, n_centers=400, bandwidth=0.5)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian(self):
    X, Y = self.get_samples()

    for method in ["agglomerative"]:
      model = KernelMixtureNetwork(center_sampling_method=method, n_centers=5)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

      p_est = model.cdf(x, y)
      p_true = norm.cdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian_noise(self):
    X, Y = self.get_samples()

    for method in ["agglomerative"]:
      model = KernelMixtureNetwork(center_sampling_method=method, n_centers=5, x_noise_std=0.2, y_noise_std=0.2)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

      p_est = model.cdf(x, y)
      p_true = norm.cdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)



  def test_MDN_with_2d_gaussian(self):
    X, Y = self.get_samples()

    model = MixtureDensityNetwork(n_centers=5)
    model.fit(X, Y)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = model.cdf(x, y)
    p_true = norm.cdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_CDE_with_2d_gaussian(self):
    X, Y = self.get_samples()

    model = ConditionalKernelDensityEstimation()
    model.fit(X, Y)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = model.cdf(x, y)
    p_true = norm.cdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


class TestConditionalDensityEstimators_fit_by_crossval(unittest.TestCase):
  def get_samples(self):
    np.random.seed(22)
    data = np.concatenate([np.random.normal([i, -i], 1, size=(500, 2)) for i in range(-20, 20, 4)], axis=0)
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_1_KMN_with_2d_gaussian_fit_by_crossval(self):
    X, Y = self.get_samples()

    param_grid = {
      "n_centers": [3, 10],
      "center_sampling_method": ["k_means"],
      "keep_edges": [True]
    }

    model = KernelMixtureNetwork(center_sampling_method="k_means", n_centers=20)
    model.fit_by_cv(X, Y, param_grid=param_grid)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertEqual(model.get_params()["n_centers"], 10)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)


  def test_2_MDN_with_2d_gaussian_fit_by_crossval(self):
    X, Y = self.get_samples()

    param_grid = {
      "n_centers": [2, 10, 50]
    }

    model = MixtureDensityNetwork()
    model.fit_by_cv(X, Y, param_grid=param_grid)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertEqual(model.get_params()["n_centers"], 10)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)


if __name__ == '__main__':
  warnings.filterwarnings("ignore")

  suite = unittest.TestSuite()
  #suite.addTest(TestConditionalDensityEstimators_2d_gaussian())
  #suite.addTest(TestConditionalDensityEstimators_fit_by_crossval())
  #suite.addTest(TestHelpers())
  suite.addTest(TestRiskMeasures())

  suite.run()