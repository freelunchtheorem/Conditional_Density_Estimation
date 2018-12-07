import sys

sys.path.append('tmp/pycharm_project_112')

from cde.evaluation.GoodnessOfFit import GoodnessOfFit, _multidim_cauchy_pdf
from cde.tests.Dummies import GaussianDummy, SimulationDummy, SkewNormalDummy
from cde.density_estimator import MixtureDensityNetwork, KernelMixtureNetwork, BaseDensityEstimator

from scipy.stats import norm
import unittest
import warnings
import scipy.stats as stats
import numpy as np

alpha = 0.05

class TestGoodnessOfFitTests(unittest.TestCase):

  """ --- plausibility checks with dummy (gaussian) estimators and simulators --- """

  # def test_gaussian_dummy_kolmogorov_cdf_1(self):
  #   est = GaussianDummy(mean=2, ndim_x=1, ndim_y=1)
  #   prob_model = SimulationDummy(mean=2, ndim_x=1, ndim_y=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=10000)
  #   gof_results = gof.compute_results()
  #   print("cdf-based KS test (t, p): ", gof_results.mean_ks_stat, gof_results.mean_ks_pval, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(gof_results.mean_ks_pval, alpha)
  #   self.assertLess(gof_results.kl_divergence, 0.01)
  #
  #
  # def test_gaussian_dummy_kolmogorov_cdf_2(self):
  #   est = GaussianDummy(mean=2)
  #   prob_model = SimulationDummy(mean=6)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=10000)
  #   gof_results = gof.compute_results()
  #   print("cdf-based KS test (t, p): ", gof_results.mean_ks_stat, gof_results.mean_ks_pval, "alpha: ", alpha)
  #   print(gof)
  #   self.assertLess(gof_results.mean_ks_pval, alpha)
  #
  #
  # def test_gaussian_dummy_kl_divergence1(self):
  #   est = GaussianDummy(mean=2)
  #   prob_model = SimulationDummy(mean=2)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=10000)
  #   kl = gof.kl_divergence()
  #   print("KL-Divergence:", kl)
  #   self.assertLess(kl, 0.2)
  #
  # def test_gaussian_dummy_kl_divergence(self):
  #   est = GaussianDummy(mean=2)
  #   prob_model = SimulationDummy(mean=4)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=10000)
  #   gof_results = gof.compute_results()
  #   print("KL-Divergence:", gof_results.kl_divergence)
  #   print(gof)
  #   self.assertGreater(gof_results.kl_divergence, 0.01)
  #
  # def test_gaussian_dummy_kl_divergence2(self):
  #   est = GaussianDummy(mean=2)
  #   prob_model1 = SimulationDummy(mean=4)
  #   gof1 = GoodnessOfFit(est, prob_model1, n_observations=10000)
  #   gof_results1 = gof1.compute_results()
  #
  #   prob_model2 = SimulationDummy(mean=3)
  #   gof2 = GoodnessOfFit(est, prob_model2, n_observations=10000)
  #   gof_results2 = gof2.compute_results()
  #   self.assertLess(gof_results2.kl_divergence, gof_results1.kl_divergence)

  def test_gaussian_dummy_hellinger_distance_mc(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2)*2
    sigma2 = np.identity(n=2)*1

    # Analytical form Hellinger Distance
    hd_squared_analytic_result = np.sqrt(1 - (2**0.5 / 1.5))

    x = np.asarray([[0, 0], [1,1]])

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x, n_mc_samples=10**7)
    self.assertAlmostEqual(hd_squared_analytic_result, gof1.hellinger_distance_mc()[0], places=2)
    self.assertAlmostEqual(hd_squared_analytic_result, gof1.hellinger_distance_mc()[1], places=2)

  def test_gaussian_dummy_kl_divergence_mc(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2) * 2
    sigma2 = np.identity(n=2) * 1

    # Analytical form Hellinger Distance
    kl_divergence_analytical = 0.5 * (1 - 2 + np.log(4))

    x = np.asarray([[0, 0], [1, 1]])

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x, n_mc_samples=10 ** 7)
    self.assertAlmostEqual(kl_divergence_analytical, gof1.kl_divergence_mc()[0], places=2)
    self.assertAlmostEqual(kl_divergence_analytical, gof1.kl_divergence_mc()[1], places=2)

  def test_gaussian_dummy_js_divergence_mc(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2) * 2
    sigma2 = np.identity(n=2) * 1

    x = np.asarray([[0, 0], [1, 1]])

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x, n_mc_samples=10 ** 7)
    self.assertGreaterEqual(1, gof1.js_divergence_mc()[0])
    self.assertLessEqual(0, gof1.js_divergence_mc()[0])

  def test_gaussian_dummy_divergence_measures_pdf(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2)*2
    sigma2 = np.identity(n=2)*1

    # Analytical form Hellinger Distance
    hd_squared_analytic_result = np.sqrt(1 - (2**0.5 / 1.5))

    # Analytical form Hellinger Distance
    kl_divergence_analytical = 0.5 * (1 - 2 + np.log(4))

    x = np.asarray([[0, 0], [1,1]])

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x, n_mc_samples=10 ** 7)

    hell_dist_mc, kl_div_mc, js_div_mc = gof1.divergence_measures_pdf()

    self.assertAlmostEqual(hd_squared_analytic_result, hell_dist_mc[0], places=2)
    self.assertAlmostEqual(kl_divergence_analytical, kl_div_mc[0], places=2)
    self.assertGreaterEqual(1, js_div_mc[0])
    self.assertLessEqual(0, js_div_mc[0])

  def test_gaussian_dummy_wasserstein_distance_mc(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2) * 2
    sigma2 = np.identity(n=2) * 1

    # Analytical form upper bound of wassersetin distance
    wasserstein_upper = np.trace(sigma1 + sigma2 - 2*(np.linalg.cholesky(sigma1.dot(sigma2))))

    x = np.asarray([[0, 0], [1, 1]])

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, n_observations=10000)
    wasserstein_dist_mc = gof1.wasserstein_distance_mc(x_cond=x)[0]
    print(wasserstein_dist_mc)
    self.assertGreaterEqual(wasserstein_upper, wasserstein_dist_mc)
    self.assertLessEqual(0, wasserstein_dist_mc)

  def test_multidim_chauchy(self):
    x1 = np.asarray([[1, 0], [0, 1]])
    x2 = np.asarray([[1]])

    self.assertEqual(_multidim_cauchy_pdf(x1)[0], _multidim_cauchy_pdf(x1)[1])
    self.assertAlmostEqual(_multidim_cauchy_pdf(x2, scale=4)[0], stats.cauchy.pdf(x2[0], scale=4)[0])

  def test_mc_intrgration_chauchy_1(self):
    mu1 = np.array([0])
    mu2 = np.array([0])
    sigma1 = np.identity(n=1) * 2
    sigma2 = np.identity(n=1) * 1

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=1, ndim_y=1)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=1, ndim_y=1)

    x_cond = np.asarray([[0]])
    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x_cond, n_mc_samples=10 ** 7)

    func = lambda y, x: stats.uniform.pdf(y, loc=-1, scale=2)

    integral = gof1._mc_integration_cauchy(func, n_samples=10**5, batch_size=10 ** 4)
    self.assertAlmostEqual(1.0, integral[0], places=2)

  def test_mc_intrgration_chauchy_2(self):
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    sigma1 = np.identity(n=2) * 2
    sigma2 = np.identity(n=2) * 1

    est = GaussianDummy(mean=mu1, cov=sigma1, ndim_x=2, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=sigma2, ndim_x=2, ndim_y=2)

    x_cond = np.asarray([[0,0]])
    gof1 = GoodnessOfFit(est, prob_model1, None, None, n_observations=10000, x_cond=x_cond, n_mc_samples=10 ** 7)

    func = lambda y, x: stats.multivariate_normal.pdf(y, mean=[0,0], cov=np.diag([2,2]))
    integral = gof1._mc_integration_cauchy(func, n_samples=10**5, batch_size=10 ** 4)
    self.assertAlmostEqual(1.0, integral[0], places=2)

class TestRiskMeasures(unittest.TestCase):

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
      CVaR_est = est.conditional_value_at_risk(x_cond=np.array([[0], [1]]), alpha=alpha)

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
    np.random.seed(22)
    from tensorflow import set_random_seed
    set_random_seed(22)

    data = np.random.normal([2, 2, 7, -2], 1, size=(5000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork("mdn_mean", 2, 2,n_centers=5)
    model.fit(X, Y)

    mean_est = model.mean_(x_cond=np.array([[1,2]]))
    self.assertAlmostEqual(mean_est[0][0], 7, places=0)
    self.assertAlmostEqual(mean_est[0][1], -2, places=1)

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

  def test_covariance_mixture(self):
    np.random.seed(22)
    from tensorflow import set_random_seed
    set_random_seed(22)

    scale = 5.0
    data = np.random.normal(loc=[2, 2, 7, -2], scale=scale, size=(5000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork("mdn_cov", 2, 2, n_centers=5)
    model.fit(X, Y)

    cov_est = model.covariance(x_cond=np.array([[0, 1]]))
    print(cov_est)
    self.assertAlmostEqual(cov_est[0][1][0], 0.0, places=1)
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
    np.random.seed(24)
    X, Y = self.get_samples(std=2)
    model = KernelMixtureNetwork("kmn-var", 1, 1, center_sampling_method="k_means", n_centers=5, n_training_epochs=50, random_seed=24)
    model.fit(X, Y)

    x_cond = np.array([[0],[1]])

    CVaR_mixture = model.conditional_value_at_risk(x_cond, alpha=0.01)
    CVaR_cdf = BaseDensityEstimator.conditional_value_at_risk(model, x_cond, alpha=0.01, n_samples=4*10**7)

    print("CVaR mixture:", CVaR_mixture)
    print("CVaR cdf:", CVaR_cdf)

    diff = np.mean(np.abs(CVaR_cdf - CVaR_mixture))
    self.assertAlmostEqual(diff, 0, places=2)

  def test_tail_risks_risk_mixture(self):
    X, Y = self.get_samples(std=2)
    model = KernelMixtureNetwork("kmn-var2", 1, 1, center_sampling_method="k_means", n_centers=5, n_training_epochs=50)
    model.fit(X, Y)

    x_cond = np.array([[0], [1]])

    VaR_mixture, CVaR_mixture = model.tail_risk_measures(x_cond, alpha=0.07)
    VaR_cdf, CVaR_mc = BaseDensityEstimator.tail_risk_measures(model, x_cond, alpha=0.07)

    print("CVaR mixture:", CVaR_mixture)
    print("CVaR cdf:", CVaR_mc)

    diff_cvar = np.mean(np.abs(CVaR_mc - CVaR_mixture))
    self.assertAlmostEqual(diff_cvar, 0, places=2)

    diff_var = np.mean(np.abs(VaR_mixture - VaR_cdf))
    self.assertAlmostEqual(diff_var, 0, places=2)


if __name__ == '__main__':
  #pytest.main('--html=unittest_report.html --self-contained-html')
  if __name__ == '__main__':
    warnings.filterwarnings("ignore")


    testmodules = [
      'unittests_evaluations.TestGoodnessOfFitTests'
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
