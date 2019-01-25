import unittest
import warnings
import numpy as np
import sys
import os
import scipy.stats as stats
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dummies import GaussianDummy, SimulationDummy, SkewNormalDummy
from cde.density_estimator import MixtureDensityNetwork, KernelMixtureNetwork, BaseDensityEstimator
from cde.evaluation.divergences import kl_divergence_pdf, js_divergence_pdf, hellinger_distance_pdf, divergence_measures_pdf

alpha = 0.05

class TestRiskMeasures(unittest.TestCase):

  def setUp(self):
    np.random.seed(23)

  def get_samples(self, std=1.0):
    np.random.seed(22)
    data = np.random.normal([2, 2], std, size=(2000, 2))
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_value_at_risk_mc(self):
    for mu, sigma in [(-2, 0.5), (0.4, 0.01), (9, 3)]:
      # prepare estimator dummy
      mu1 = np.array([mu])
      sigma1 = np.identity(n=1)*sigma
      est = GaussianDummy(mean=mu1, cov=sigma1**2, ndim_x=1, ndim_y=1, has_cdf=False)
      est.fit(None, None)

      alpha = 0.05
      VaR_est = est.value_at_risk(x_cond=np.array([[0],[1]]), alpha=alpha)
      VaR_true = norm.ppf(alpha, loc=mu, scale=sigma)
      self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
      self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_value_at_risk_cdf(self):
    for mu, sigma in [(-2, 0.5), (0.4, 0.01), (22, 3)]:
      # prepare estimator dummy
      mu1 = np.array([mu])
      sigma1 = np.identity(n=1)*sigma
      est = GaussianDummy(mean=mu1, cov=sigma1**2, ndim_x=1, ndim_y=1, has_cdf=True)
      est.fit(None, None)

      alpha = 0.05
      VaR_est = est.value_at_risk(x_cond=np.array([[0],[1]]), alpha=alpha)
      VaR_true = norm.ppf(alpha, loc=mu, scale=sigma)
      self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
      self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_conditional_value_at_risk_mc(self):
    for mu, sigma, alpha in [(1, 1, 0.05), (0.4, 0.1, 0.02), (0.1, 2, 0.01)]:
      # prepare estimator dummy
      mu1 = np.array([mu])
      sigma1 = np.identity(n=1) * sigma
      est = GaussianDummy(mean=mu1, cov=sigma1**2, ndim_x=1, ndim_y=1, has_pdf=True)
      est.fit(None, None)

      CVaR_true = mu - sigma/alpha * norm.pdf(norm.ppf(alpha))
      CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0],[1]]), alpha=alpha)

      print("CVaR True (%.2f, %.2f):"%(mu, sigma), CVaR_true)
      print("CVaR_est (%.2f, %.2f):"%(mu, sigma), CVaR_est)
      print("VaR (%.2f, %.2f):"%(mu, sigma), est.value_at_risk(x_cond=np.array([[0],[1]]), alpha=alpha))

      self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
      self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_conditional_value_at_risk_sample(self):
    # prepare estimator dummy
    for mu, sigma in [(-6, 0.25), (0.4, 0.1), (22, 3)]:
      mu1 = np.array([mu])
      sigma1 = np.identity(n=1) * sigma
      est = GaussianDummy(mean=mu1, cov=sigma1**2, ndim_x=1, ndim_y=1, has_pdf=False)
      est.fit(None, None)

      alpha = 0.02

      CVaR_true = mu - sigma / alpha * norm.pdf(norm.ppf(alpha))
      CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0], [1]]), alpha=alpha, n_samples=2*10**6)

      self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
      self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_mean_mc(self):
    # prepare estimator dummy
    mu = np.array([0,1])
    sigma = np.identity(n=2) * 1
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, has_cdf=False)
    est.fit(None, None)

    mean_est = est.mean_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)

  def test_mean_pdf(self):
    # prepare estimator dummy
    mu = np.array([0, 1])
    sigma = np.identity(n=2) * 1
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)
    est.fit(None, None)

    mean_est = est.mean_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)

  def test_mean_mixture(self):
    np.random.seed(24)
    from tensorflow import set_random_seed
    set_random_seed(24)

    data = np.random.normal([2, 2, 7, -2], 1, size=(5000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork("mdn_mean", 2, 2, n_centers=3, y_noise_std=0.1, x_noise_std=0.1)
    model.fit(X, Y)

    mean_est = model.mean_(x_cond=np.array([[1.5, 2]]), n_samples=10**7)
    self.assertAlmostEqual(mean_est[0][0], 7, places=0)
    self.assertAlmostEqual(mean_est[0][1], -2, places=0)

  def test_std1(self):
    # prepare estimator dummy
    mu = np.array([0, 1])
    sigma = np.array([[1, -0.2], [-0.2, 2]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)
    est.fit(None, None)

    std = est.std_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(std[0][0], np.sqrt(sigma[0][0]), places=2)
    self.assertAlmostEqual(std[0][1], np.sqrt(sigma[1][1]), places=2)

  def test_std2(self):
    # prepare estimator dummy
    mu = np.array([14])
    sigma = np.array([[0.1]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=1, ndim_y=1, can_sample=False)
    est.fit(None, None)

    std_est = est.std_(x_cond=np.array([[0.0], [1.0]]))
    self.assertAlmostEqual(std_est[0][0]**2, sigma[0][0], places=2)
    self.assertAlmostEqual(std_est[1][0]**2, sigma[0][0], places=2)

  def test_covariance1(self):
    # prepare estimator dummy
    mu = np.array([-1])
    sigma = np.array([[0.1]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=1)
    est.fit(None, None)

    cov_est = est.covariance(x_cond=np.array([[0.5, 2]]))
    self.assertAlmostEqual(cov_est[0][0][0], sigma[0][0], places=2)

  def test_covariance2(self):
    # prepare estimator dummy
    mu = np.array([0, 1])
    sigma = np.array([[1,-0.2],[-0.2,2]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)
    est.fit(None, None)

    cov_est = est.covariance(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(cov_est[0][0][0], sigma[0][0], places=2)
    self.assertAlmostEqual(cov_est[0][1][0], sigma[1][0], places=2)

  def test_mean_std(self):
    mu = np.array([0, 1])
    sigma = np.array([[1, -0.2], [-0.2, 2]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)
    est.fit(None, None)

    mean_est, std_est = est.mean_std(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)
    self.assertAlmostEqual(std_est[0][0]**2, sigma[0][0], places=2)
    self.assertAlmostEqual(std_est[0][1]**2, sigma[1][1], places=2)

  def test_covariance_mixture(self):
    np.random.seed(24)
    from tensorflow import set_random_seed
    set_random_seed(24)

    scale = 2.0
    data = np.random.normal(loc=[2, 2, 7, -2], scale=scale, size=(10000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork("mdn_cov", 2, 2, n_centers=5, x_noise_std=0.1, y_noise_std=0.1)
    model.fit(X, Y)

    cov_est = model.covariance(x_cond=np.array([[0, 1]]))
    print(cov_est)
    self.assertLessEqual(np.abs(cov_est[0][1][0] - 0.0), 0.2)
    self.assertLessEqual(np.abs(np.sqrt(cov_est[0][0][0]) - scale),1.0)

  def test_skewness1(self):
    mu = np.array([-0.001])
    sigma = np.array([[0.02]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=1, can_sample=False)
    est.fit(None, None)

    skew = est._skewness_mc(x_cond=np.array([[0, 1]]))
    print("Skewness sample estimate:", skew)
    self.assertAlmostEqual(skew[0], 0, places=1)

    skew = est._skewness_pdf(x_cond=np.array([[0, 1]]))
    print("Skewness pdf estimate:", skew)
    self.assertAlmostEqual(skew[0], 0, places=1)

  def test_skewness2(self):
    est = SkewNormalDummy(shape=-2, ndim_x=2, ndim_y=1)
    est.fit(None, None)

    skew = est._skewness_mc(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Skewness sample estimate:", skew)
    self.assertAlmostEqual(skew[0], est.skewness, places=1)

    skew = est._skewness_pdf(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Skewness pdf estimate:", skew)
    self.assertAlmostEqual(skew[0], est.skewness, places=1)

    print("True Skewness value", est.skewness)

  def test_kurtosis1(self):
    mu = np.array([-0.001])
    sigma = np.array([[0.02]])
    est = GaussianDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=1, can_sample=False)
    est.fit(None, None)

    kurt = est._kurtosis_mc(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Kurtosis sample estimate:", kurt)
    self.assertAlmostEqual(kurt[0], 0, places=1)

    kurt = est._kurtosis_pdf(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Kurtosis pdf estimate:", kurt)
    self.assertAlmostEqual(kurt[0], 0, places=1)

  def test_kurtosis2(self):
    est = SkewNormalDummy(shape=-2, ndim_x=2, ndim_y=1)
    est.fit(None, None)

    kurt = est._kurtosis_mc(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Kurtosis sample estimate:", kurt)
    self.assertAlmostEqual(kurt[0], est.kurtosis, places=1)

    kurt = est._kurtosis_pdf(x_cond=np.array([[0, 1]]), n_samples=10**6)
    print("Kurtosis pdf estimate:", kurt)
    self.assertAlmostEqual(kurt[0], est.kurtosis, places=1)

    print("True Kurtosis value", est.kurtosis)

  def test_conditional_value_at_risk_mixture(self):
    np.random.seed(20)
    X, Y = self.get_samples(std=0.5)
    model = KernelMixtureNetwork("kmn-var", 1, 1, center_sampling_method="k_means", n_centers=5, n_training_epochs=500, random_seed=24)
    model.fit(X, Y)

    x_cond = np.array([[0],[1]])

    CVaR_mixture = model.conditional_value_at_risk(x_cond, alpha=0.05)
    CVaR_cdf = BaseDensityEstimator.conditional_value_at_risk(model, x_cond, alpha=0.05, n_samples=5*10**7)

    print("CVaR mixture:", CVaR_mixture)
    print("CVaR cdf:", CVaR_cdf)

    diff = np.mean(np.abs(CVaR_cdf - CVaR_mixture))
    self.assertAlmostEqual(diff, 0, places=1)

  def test_tail_risks_risk_mixture(self):
    X, Y = self.get_samples(std=0.5)
    model = KernelMixtureNetwork("kmn-var2", 1, 1, center_sampling_method="k_means", n_centers=5, n_training_epochs=50)
    model.fit(X, Y)

    x_cond = np.array([[0], [1]])

    VaR_mixture, CVaR_mixture = model.tail_risk_measures(x_cond, alpha=0.07)
    VaR_cdf, CVaR_mc = BaseDensityEstimator.tail_risk_measures(model, x_cond, alpha=0.07)

    print("CVaR mixture:", CVaR_mixture)
    print("CVaR cdf:", CVaR_mc)

    diff_cvar = np.mean(np.abs(CVaR_mc - CVaR_mixture))
    self.assertAlmostEqual(diff_cvar, 0, places=1)

    diff_var = np.mean(np.abs(VaR_mixture - VaR_cdf))
    self.assertAlmostEqual(diff_var, 0, places=1)

class TestDivergenceMeasures(unittest.TestCase):

  def setUp(self):
    self.mu1 = np.array([0.0])
    self.cov1 = np.eye(1)
    self.mu2 = np.array([1.0])
    self.cov2 = np.eye(1) * 2

    self.gaussian1 = GaussianDummy(mean=self.mu1, cov=self.cov1, ndim_x=2, ndim_y=1)
    self.gaussian2 = GaussianDummy(mean=self.mu2, cov=self.cov2, ndim_x=2, ndim_y=1)

    self.mu3 = np.array([1.6, -7.0])
    self.cov3 = np.array([[2.0, 0.5], [0.5, 4.5]])
    self.mu4 = np.array([1.0, -5.0])
    self.cov4 = np.eye(2) * 2

    self.gaussian3 = GaussianDummy(mean=self.mu3, cov=self.cov3, ndim_x=2, ndim_y=2)
    self.gaussian4 = GaussianDummy(mean=self.mu4, cov=self.cov4, ndim_x=2, ndim_y=2)

  def test_kl_gaussian(self):
    kl1 = _kl_gaussians(self.mu1, self.cov1, self.mu1, self.cov1)
    self.assertAlmostEqual(float(kl1), 0.0)

    kl2 = _kl_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)
    self.assertGreater(float(kl2), 0.0)

    mu = np.array([1.6, -7.0])
    cov = np.array([[2.0, 0.5], [0.5, 4.5]])
    kl3 = _kl_gaussians(mu, cov, mu, cov)
    self.assertAlmostEqual(float(kl3), 0.0)

  def test_hellinger_gaussian(self):
    kl1 = _hellinger_gaussians(self.mu1, self.cov1, self.mu1, self.cov1)
    self.assertAlmostEqual(float(kl1), 0.0)

    kl2 = _hellinger_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)
    self.assertGreater(float(kl2), 0.0)

    mu = np.array([1.6, -7.0])
    cov = np.array([[2.0, 0.5], [0.5, 4.5]])
    kl3 = _hellinger_gaussians(mu, cov, mu, cov)
    self.assertAlmostEqual(float(kl3), 0.0)

  def test_kl_mc_1d(self):
    x_cond = np.array([[0.0, 1.0]])
    kl_est = kl_divergence_pdf(self.gaussian1, self.gaussian2, x_cond=x_cond)
    kl_true = _kl_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)
    print(kl_est[0], kl_true)
    self.assertAlmostEqual(kl_est[0], kl_true, places=1)

  def test_kl_mc_2d(self):
    x_cond = np.array([[0.0, 1.0]])
    kl_est = kl_divergence_pdf(self.gaussian3, self.gaussian4, x_cond=x_cond)
    kl_true = _kl_gaussians(self.mu3, self.cov3, self.mu4, self.cov4)
    print(kl_est[0], kl_true)
    self.assertAlmostEqual(kl_est[0], kl_true, places=1)

  def test_hellinger_mc_1d(self):
    x_cond = np.array([[0.0, 1.0]])
    h_est = hellinger_distance_pdf(self.gaussian1, self.gaussian2, x_cond=x_cond)
    h_true = _hellinger_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)
    print(h_est[0], h_true)
    self.assertAlmostEqual(h_est[0], h_true, places=1)

  def test_hellinger_mc_2d(self):
    x_cond = np.array([[0.0, 1.0]])
    h_est = hellinger_distance_pdf(self.gaussian3, self.gaussian4, x_cond=x_cond)
    h_true = _hellinger_gaussians(self.mu3, self.cov3, self.mu4, self.cov4)
    print(h_est[0], h_true)
    self.assertAlmostEqual(h_est[0], h_true, places=1)

  def test_js_mc_1d(self):
    x_cond = np.array([[0.0, 1.0]])
    js_est = js_divergence_pdf(self.gaussian1, self.gaussian2, x_cond=x_cond)
    js_true = 0.5 * _kl_gaussians(self.mu1, self.cov1, self.mu2, self.cov2) + 0.5 * _kl_gaussians(self.mu2, self.cov2, self.mu1, self.cov1)
    print(js_est[0], js_true)
    self.assertAlmostEqual(js_est[0], js_true, places=1)

  def test_js_mc_2d(self):
    x_cond = np.array([[0.0, 1.0]])
    kl_est = js_divergence_pdf(self.gaussian3, self.gaussian4, x_cond=x_cond)
    kl_true = 0.5 * _kl_gaussians(self.mu3, self.cov3, self.mu4, self.cov4) + 0.5 * _kl_gaussians(self.mu4, self.cov4, self.mu3, self.cov3)
    print(kl_est[0], kl_true)
    self.assertAlmostEqual(kl_est[0], kl_true, places=1)

  def test_divmeasures_mc_1d(self):
    np.random.seed(22)
    x_cond = np.array([[0.0, 1.0], [2.0, 0.5]])
    h_est, kl_est, js_est = divergence_measures_pdf(self.gaussian1, self.gaussian2, x_cond=x_cond)

    js_true = 0.5 * _kl_gaussians(self.mu1, self.cov1, self.mu2, self.cov2) + 0.5 * _kl_gaussians(self.mu2, self.cov2, self.mu1, self.cov1)
    h_true = _hellinger_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)
    kl_true = _kl_gaussians(self.mu1, self.cov1, self.mu2, self.cov2)

    self.assertAlmostEqual(js_est[1], js_true, places=1)
    self.assertAlmostEqual(kl_est[0], kl_true, places=1)
    self.assertAlmostEqual(h_est[0], h_true, places=1)

  def test_divmeasures_mc_2d(self):
    np.random.seed(22)
    x_cond = np.array([[0.0, 1.0]])
    h_est, kl_est, js_est = divergence_measures_pdf(self.gaussian3, self.gaussian4, x_cond=x_cond)

    js_true = 0.5 * _kl_gaussians(self.mu3, self.cov3, self.mu4, self.cov4) + 0.5 * _kl_gaussians(self.mu4, self.cov4,
                                                                                                  self.mu3, self.cov3)
    h_true = _hellinger_gaussians(self.mu3, self.cov3, self.mu4, self.cov4)
    kl_true = _kl_gaussians(self.mu3, self.cov3, self.mu4, self.cov4)

    self.assertAlmostEqual(js_est[0], js_true, places=1)
    self.assertAlmostEqual(kl_est[0], kl_true, places=1)
    self.assertAlmostEqual(h_est[0], h_true, places=1)

def _kl_gaussians(mu1, cov1, mu2, cov2):
  assert cov1.shape == cov2.shape
  assert mu1.shape == mu2.shape
  term1 = np.log(np.linalg.det(cov2)) - np.log(np.linalg.det(cov1))
  term2 = np.trace(np.linalg.inv(cov2).dot(cov1))
  term3 = np.transpose(mu2 - mu1).dot(np.linalg.inv(cov2)).dot(mu2-mu1)
  return 0.5 * (term1 - cov1.shape[0] + term2 + term3)

def _hellinger_gaussians(mu1, cov1, mu2, cov2):
  assert cov1.shape == cov2.shape
  assert mu1.shape == mu2.shape
  term1 = np.linalg.det(cov1)**0.25 * np.linalg.det(cov2)**0.25
  term2 = np.linalg.det(0.5 * (cov1 + cov2))**0.5
  term3 = np.exp(-0.125 * np.transpose(mu2 - mu1).dot(np.linalg.inv(0.5 * (cov1 + cov2))).dot(mu2-mu1))
  return np.sqrt(1 - term1 / term2 * term3)

if __name__ == '__main__':
  warnings.filterwarnings("ignore")

  testmodules = [
    'unittests_evaluations.TestDivergenceMeasures',
    'unittests_evaluations.TestRiskMeasures',
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
