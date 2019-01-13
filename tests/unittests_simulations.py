import unittest
import sys
import os
import scipy.stats as stats
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
from cde.density_simulation import SkewNormal, GaussianMixture, EconDensity, JumpDiffusionModel, ArmaJump
from cde.helpers import mc_integration_cauchy
from .Dummies import SimulationDummy


class TestArmaJump(unittest.TestCase):

  def test_cdf_sample_consistency(self):
    arj = ArmaJump(random_seed=8787)

    x_cond = np.asarray([0.1 for _ in range(200000)])
    _, y_sample = arj.simulate_conditional(x_cond)


    cdf_callable = lambda y: arj.cdf(x_cond, y)
    _, p_val = stats.kstest(y_sample, cdf_callable)
    print("P-Val Kolmogorov:", p_val)

    self.assertGreaterEqual(p_val, 0.5)

  def test_skewness(self):
    np.random.seed(22)
    arj = ArmaJump(jump_prob=0.01)
    x_cond = np.asarray([0.1 for _ in range(200000)])
    _, y_sample = arj.simulate_conditional(x_cond)

    skew1 = stats.skew(y_sample)

    arj = ArmaJump(jump_prob=0.1)
    _, y_sample = arj.simulate_conditional(x_cond)

    skew2 = stats.skew(y_sample)

    print(skew1, skew2)
    self.assertLessEqual(skew2, skew1)

  def test_mean(self):
    np.random.seed(22)
    arj = ArmaJump(c=0.1, jump_prob=0.00)
    x_cond = np.asarray([0.1])
    mean = arj.mean_(x_cond)
    self.assertAlmostEqual(mean, 0.1)

    arj = ArmaJump(c=0.1, jump_prob=0.1)
    mean = arj.mean_(x_cond)
    self.assertLessEqual(mean, 0.1)

  def test_cov(self):
    np.random.seed(22)
    arj = ArmaJump(c=0.1, jump_prob=0.00, std=0.1)
    x_cond = np.asarray([0.1])
    cov = arj.covariance(x_cond)[0][0][0]
    self.assertAlmostEqual(cov, 0.1**2)

    arj = ArmaJump(c=0.1, jump_prob=0.1, std=0.1)
    cov = arj.covariance(x_cond)[0][0][0]
    self.assertGreater(cov, 0.1 ** 2)

class TestGaussianMixture(unittest.TestCase):

  def test_mean(self):
    gmm = GaussianMixture(n_kernels=5, random_seed=24, ndim_x=2, ndim_y=2)

    x_cond = np.array([[1.0,1.0]])
    mean_mc = mean_pdf(gmm, x_cond)

    mean = gmm.mean_(x_cond).flatten()

    print(mean_mc)
    print(mean)

    self.assertLessEqual(np.sum((mean_mc - mean)**2), 0.1)

  def test_covariance(self):
    gmm = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)
    x_cond = np.array([[1.0, 1.0]])
    cov = gmm.covariance(x_cond)

    cov_mc = covariance_pdf(gmm, x_cond)

    print(cov[0])
    print(cov_mc[0])
    self.assertLessEqual(np.sum((cov_mc - cov) ** 2), 0.1)

  def test_sampling(self):
    gmm = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)

    ## simulate unconditionally
    x, y = gmm.simulate(n_samples=10 ** 6)

    # simulte conditionally
    x_cond = 2 * np.ones(shape=(10 ** 6, 2))
    _,  y_sample = gmm.simulate_conditional(x_cond)
    self.assertLessEqual(np.mean(np.abs(gmm.mean_(x_cond)[0] - y_sample.mean(axis=0))), 0.1)

  def test_sampling_consistency(self):
    from cde.evaluation.ConfigRunner import make_hash_sha256
    gmm1 = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)
    x1, y1 = gmm1.simulate(n_samples=10 ** 3)
    hash_x1 = make_hash_sha256(x1)
    hash_y1 = make_hash_sha256(y1)

    gmm2 = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)
    x2, y2 = gmm2.simulate(n_samples=10 ** 3)

    hash_x2 = make_hash_sha256(x2)
    hash_y2 = make_hash_sha256(y2)

    self.assertEqual(hash_x1, hash_x2)
    self.assertEqual(hash_y1, hash_y2)

  def test_hash(self):
    from cde.evaluation.ConfigRunner import make_hash_sha256
    gmm1 = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)
    gmm2 = GaussianMixture(n_kernels=2, random_seed=54, ndim_x=2, ndim_y=2)
    hash1 = make_hash_sha256(gmm1)
    hash2 = make_hash_sha256(gmm2)
    self.assertEqual(hash1, hash2)

class TestEconDensity(unittest.TestCase):
  def test_pdf(self):
    sim_model = EconDensity()
    x = np.ones(shape=(2000,1))
    y = np.random.uniform(0.01, 5, size=(2000,1))
    p_sim = sim_model.pdf(x,y)
    p_true = stats.norm.pdf(y, loc=1, scale=2)
    diff = np.sum(np.abs(p_sim - p_true))
    self.assertAlmostEquals(diff, 0.0, places=2)

  def test_cdf(self):
    sim_model = EconDensity()
    x = np.ones(shape=(2000,1))
    y = np.random.uniform(0.01, 5, size=(2000,1))
    p_sim = sim_model.cdf(x,y)
    p_true = stats.norm.cdf(y, loc=1, scale=2)
    diff = np.sum(np.abs(p_sim - p_true))
    self.assertAlmostEquals(diff, 0.0, places=2)

  def test_value_at_risk(self):
    sim_model = EconDensity()
    x_cond = np.array([[0], [1]])
    VaR = sim_model.value_at_risk(x_cond, alpha=0.05)

    VaR_cdf = super(EconDensity, sim_model).value_at_risk(x_cond, alpha=0.05)

    diff = np.sum(np.abs(VaR_cdf - VaR))

    self.assertAlmostEqual(VaR[0], stats.norm.ppf(0.05, loc=0, scale=1), places=4)
    self.assertAlmostEqual(VaR[1], stats.norm.ppf(0.05, loc=1, scale=2), places=4)
    self.assertAlmostEqual(diff, 0, places=4)

  def test_conditional_value_at_risk(self):
    sim_model = EconDensity()
    x_cond = np.array([[0], [1]])
    CVaR = sim_model.conditional_value_at_risk(x_cond, alpha=0.03)

    CVaR_mc = super(EconDensity, sim_model).conditional_value_at_risk(x_cond, alpha=0.03, n_samples=10**7)

    print("CVaR Analytic:", CVaR)
    print("CVaR MC:", CVaR_mc)
    print("VaR", sim_model.value_at_risk(x_cond, alpha=0.03))

    diff = np.mean(np.abs(CVaR_mc - CVaR))

    self.assertAlmostEqual(diff, 0, places=2)

  def test_random_seed(self):
    sim_model1 = EconDensity(random_seed=22)
    X1, Y1 = sim_model1.simulate(n_samples=100)

    sim_model2 = EconDensity(random_seed=22)
    X2, Y2 = sim_model2.simulate(n_samples=100)

    diff_x = np.sum(np.abs(X1[:100] - X2[:]))
    diff_y = np.sum(np.abs(Y1[:100] - Y2[:]))
    self.assertAlmostEquals(diff_x, 0, places=2)
    self.assertAlmostEquals(diff_y, 0, places=2)

class TetsSkewNormal(unittest.TestCase):
  def setUp(self):
    self.dist = SkewNormal(random_seed=22)

  def test_pdf(self):
    X = np.linspace(-1, 1, num=1000)
    Y = np.linspace(-1, 1, num=1000)
    p = self.dist.pdf(X, Y)
    self.assertEqual(p.shape, (1000,))

  def test_cdf(self):
    X = np.linspace(-1, 1, num=1000)
    Y = np.linspace(-1, 1, num=1000)
    p = self.dist.cdf(X, Y)
    self.assertEqual(p.shape, (1000,))

  def test_simulate_conditional_skew(self):
    x_cond_pos = np.ones(10**4) * 0.1
    y_samples_pos = self.dist.simulate_conditional(x_cond_pos)
    self.assertEqual(y_samples_pos.shape, (int(10**4), 1))
    x_cond_neg = - np.ones(10 ** 4) * 0.1
    y_samples_neg = self.dist.simulate_conditional(x_cond_neg)
    self.assertEqual(y_samples_neg.shape, (int(10**4), 1))

    skew_pos = stats.skew(y_samples_pos)
    skew_neg = stats.skew(y_samples_neg)
    self.assertLess(skew_neg, skew_pos)

  def test_simulate(self):
    x, y = self.dist.simulate(n_samples=1000)
    self.assertEqual(x.shape, (1000, 1))
    self.assertEqual(y.shape, (1000, 1))

  def test_seed(self):
    sim_1 = SkewNormal(random_seed=22)
    x1 = sim_1.simulate(100)
    sim_2 = SkewNormal(random_seed=22)
    x2 = sim_2.simulate(100)
    self.assertTrue(np.allclose(x1, x2))

class TestJumpDiffusionModel(unittest.TestCase):

  def test_simulate_on_skewness(self):
    np.random.seed(22)
    jdm = JumpDiffusionModel()
    _, y = jdm.simulate(n_samples=10000)
    skew = stats.skew(y)
    self.assertLessEqual(skew, -0.5)

  def test_simulate_conditional_on_skewness(self):
    np.random.seed(22)
    jdm = JumpDiffusionModel()
    x, y = jdm.simulate(n_samples=10000)
    x_cond = np.tile(np.expand_dims(x[5], axis=0), (10000, 1))
    _, y2 = jdm.simulate_conditional(x_cond)

    skew = stats.skew(y2)
    self.assertLessEqual(skew, -0.5)

  def test_mean(self):
    np.random.seed(22)
    jdm = JumpDiffusionModel()
    x_cond = np.array([[jdm.V_0, jdm.L_0, jdm.Psi_0]])
    mean = jdm.mean_(x_cond)[0][0]
    self.assertAlmostEqual(mean, 0.0, places=2)

  def test_covariance(self):
    np.random.seed(22)
    jdm = JumpDiffusionModel()
    x_cond = np.array([[jdm.V_0, jdm.L_0, jdm.Psi_0]])
    cov = jdm.covariance(x_cond)[0][0][0]
    self.assertAlmostEqual(cov, 0.0, places=2)

  def test_VaR(self):
    np.random.seed(22)
    jdm = JumpDiffusionModel()
    x_cond = np.array([[jdm.V_0, jdm.L_0, jdm.Psi_0]])
    VaR = jdm.value_at_risk(x_cond)[0]
    self.assertLessEqual(VaR, -0.01)

class TestRiskMeasures(unittest.TestCase):
  def test_value_at_risk_mc(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.01
    VaR_est = est.value_at_risk(x_cond=np.array([[0], [1]]), alpha=alpha)
    VaR_true = stats.norm.ppf(alpha, loc=0, scale=1)
    self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
    self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_value_at_risk_cdf(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=True)

    alpha = 0.05
    VaR_est = est.value_at_risk(x_cond=np.array([[0], [1]]), alpha=alpha)
    VaR_true = stats.norm.ppf(alpha, loc=0, scale=1)
    self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
    self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_conditional_value_at_risk_mc(self):
    # prepare estimator dummy
    np.random.seed(22)
    mu = 0
    sigma = 1
    mu1 = np.array([mu])
    sigma1 = np.identity(n=1) * sigma
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=True)

    alpha = 0.02

    CVaR_true = mu - sigma/alpha * stats.norm.pdf(stats.norm.ppf(alpha, loc=0, scale=1))
    CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0], [1]]), alpha=alpha, n_samples=10**7)

    self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
    self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_conditional_value_at_risk_mc_2dim_xcond(self):
    # prepare estimator dummy
    mu = 0
    sigma = 1
    mu1 = np.array([mu])
    sigma1 = np.identity(n=1) * sigma
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=1, has_cdf=False)

    alpha = 0.02
    # x_cond shape (2,2)
    CVaR_true = mu - sigma/alpha * stats.norm.pdf(stats.norm.ppf(alpha, loc=0, scale=1))
    CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0, 1], [0, 1]]), alpha=alpha, n_samples=10**8)

    self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
    self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_conditional_value_at_risk_mc_1dim_xcond_flattend(self):
    # prepare estimator dummy
    np.random.seed(22)
    mu = 0
    sigma = 1
    mu1 = np.array([mu])
    sigma1 = np.identity(n=1) * sigma
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.02

    # x_cond shape (2,)
    CVaR_true = mu - sigma / alpha * stats.norm.pdf(stats.norm.ppf(alpha, loc=0, scale=1))
    CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0], [1]]).flatten(), alpha=alpha, n_samples=4*10**7)

    self.assertAlmostEqual(CVaR_est[0], CVaR_true, places=2)
    self.assertAlmostEqual(CVaR_est[1], CVaR_true, places=2)

  def test_mean_mc(self):
    # prepare estimator dummy
    mu = np.array([0,1])
    sigma = np.identity(n=2) * 1
    est = SimulationDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, has_cdf=False)

    mean_est = est.mean_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)

  def test_mean_pdf(self):
    # prepare estimator dummy
    mu = np.array([0, 1])
    sigma = np.identity(n=2) * 1
    est = SimulationDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)

    mean_est = est.mean_(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(mean_est[0][0], mu[0], places=2)
    self.assertAlmostEqual(mean_est[0][1], mu[1], places=2)

  def test_covariance(self):
    # prepare estimator dummy
    mu = np.array([0, 1])
    sigma = np.array([[1,-0.2],[-0.2,2]])
    est = SimulationDummy(mean=mu, cov=sigma, ndim_x=2, ndim_y=2, can_sample=False)

    cov_est = est.covariance(x_cond=np.array([[0, 1]]))
    self.assertAlmostEqual(cov_est[0][0][0], sigma[0][0], places=2)
    self.assertAlmostEqual(cov_est[0][1][0], sigma[1][0], places=2)


def mean_pdf(density, x_cond, n_samples=10 ** 6):
  means = np.zeros((x_cond.shape[0], density.ndim_y))
  for i in range(x_cond.shape[0]):
    x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
    func = lambda y: y * np.tile(np.expand_dims(density.pdf(x, y), axis=1), (1, density.ndim_y))
    integral = mc_integration_cauchy(func, ndim=2, n_samples=n_samples)
    means[i] = integral
  return means

def covariance_pdf(density, x_cond, n_samples=10 ** 6):
  covs = np.zeros((x_cond.shape[0], density.ndim_y, density.ndim_y))
  mean = density.mean_(x_cond)
  for i in range(x_cond.shape[0]):
    x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))

    def cov(y):
      a = (y - mean[i])

      #compute cov matrices c for sampled instances and weight them with the probability p from the pdf
      c = np.empty((a.shape[0], a.shape[1]**2))
      for j in range(a.shape[0]):
        c[j,:] = np.outer(a[j],a[j]).flatten()

      p = np.tile(np.expand_dims(density.pdf(x, y), axis=1), (1, density.ndim_y ** 2))
      res = c * p
      return res

    integral = mc_integration_cauchy(cov, ndim=density.ndim_y, n_samples=n_samples)
    covs[i] = integral.reshape((density.ndim_y, density.ndim_y))
  return covs


if __name__ == '__main__':

    testmodules = [
      'unittests_simulations.TestArmaJump',
      'unittests_simulations.TestGaussianMixture',
      'unittests_simulations.TestEconDensity',
      'unittests_simulations.TestJumpDiffusionModel',
      'unittests_simulations.TestRiskMeasures'
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
