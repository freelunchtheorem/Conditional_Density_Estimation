import unittest
from cde.density_simulation import *
from cde.helpers import *
from cde.tests import SimulationDummy
from scipy.stats import norm

class TestArmaJump(unittest.TestCase):

  def test_cdf_sample_consistency(self):
    np.random.seed(8787)
    arj = ArmaJump()

    x_cond = np.asarray([0.1 for _ in range(200000)])
    y_sample = arj.simulate_conditional(x_cond)


    cdf_callable = lambda y: arj.cdf(x_cond, y)
    _, p_val = stats.kstest(y_sample, cdf_callable)
    print("P-Val Kolmogorov:", p_val)

    self.assertGreaterEqual(p_val, 0.5)

  def test_skewness(self):
    np.random.seed(22)
    arj = ArmaJump(jump_prob=0.01)
    x_cond = np.asarray([0.1 for _ in range(200000)])
    y_sample = arj.simulate_conditional(x_cond)

    skew1 = stats.skew(y_sample)

    arj = ArmaJump(jump_prob=0.1)
    y_sample = arj.simulate_conditional(x_cond)

    skew2 = stats.skew(y_sample)

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

class TestRiskMeasures(unittest.TestCase):
  def test_value_at_risk_mc(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.01
    VaR_est = est.value_at_risk(x_cond=np.array([0,1]), alpha=alpha)
    VaR_true = norm.ppf(alpha, loc=0, scale=1)
    self.assertAlmostEqual(VaR_est[0], VaR_true, places=2)
    self.assertAlmostEqual(VaR_est[1], VaR_true, places=2)

  def test_value_at_risk_cdf(self):
    # prepare estimator dummy
    mu1 = np.array([0])
    sigma1 = np.identity(n=1)*1
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=True)

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
    est = SimulationDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1, has_cdf=False)

    alpha = 0.02

    CVaR_true = mu - sigma/alpha * norm.pdf(norm.ppf(alpha, loc=0, scale=1))
    CVaR_est = est.conditional_value_at_risk(x_cond=np.array([0, 1]), alpha=alpha)

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