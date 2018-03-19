import unittest
import pytest
import scipy.stats as stats
import numpy as np
from cde.density_simulation import EconDensity, GaussianMixture
from cde.evaluation.GoodnessOfFit import GoodnessOfFit
from cde.tests.Dummies import GaussianDummy, SimulationDummy

alpha = 0.05

class TestGoodnessOfFitTests(unittest.TestCase):

  """ --- plausibility checks with dummy (gaussian) estimators and simulators --- """

  def test_gaussian_dummy_kolmogorov_cdf_1(self):
    est = GaussianDummy(mean=2, ndim_x=1, ndim_y=1)
    prob_model = SimulationDummy(mean=2, ndim_x=1, ndim_y=1)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    gof_results = gof.compute_results()
    print("cdf-based KS test (t, p): ", gof_results.mean_ks_stat, gof_results.mean_ks_pval, "alpha: ", alpha)
    print(gof)
    self.assertGreater(gof_results.mean_ks_pval, alpha)
    self.assertLess(gof_results.kl_divergence, 0.01)


  def test_gaussian_dummy_kolmogorov_cdf_2(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=6)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    gof_results = gof.compute_results()
    print("cdf-based KS test (t, p): ", gof_results.mean_ks_stat, gof_results.mean_ks_pval, "alpha: ", alpha)
    print(gof)
    self.assertLess(gof_results.mean_ks_pval, alpha)


  def test_gaussian_dummy_kl_divergence1(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=2)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    kl = gof.kl_divergence()
    print("KL-Divergence:", kl)
    self.assertLess(kl, 0.2)

  def test_gaussian_dummy_kl_divergence(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=4)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    gof_results = gof.compute_results()
    print("KL-Divergence:", gof_results.kl_divergence)
    print(gof)
    self.assertGreater(gof_results.kl_divergence, 0.01)

  def test_gaussian_dummy_kl_divergence2(self):
    est = GaussianDummy(mean=2)
    prob_model1 = SimulationDummy(mean=4)
    gof1 = GoodnessOfFit(est, prob_model1, n_observations=10000)
    gof_results1 = gof1.compute_results()

    prob_model2 = SimulationDummy(mean=3)
    gof2 = GoodnessOfFit(est, prob_model2, n_observations=10000)
    gof_results2 = gof2.compute_results()
    self.assertLess(gof_results2.kl_divergence, gof_results1.kl_divergence)

  def test_gaussian_dummy_hellinger_distance(self):
    mu1 = np.array([1, 2])
    mu2 = np.array([1, 3])
    dev1 = np.identity(n=2)*4
    dev2 = np.identity(n=2)*3

    numerator = np.linalg.det(dev1)**(1/4) * np.linalg.det(dev2)**(1/4)
    denominator = np.linalg.det((dev1+dev2)/2)**(1/2)
    exponent = (-1/8 * (mu1-mu2).T * np.linalg.inv((dev1+dev2)/2) * (mu1-mu2))

    hd_squared_analytic_result = 1 - (numerator/denominator) * np.exp(exponent)

    est = GaussianDummy(mean=mu1, cov=dev1, ndim_y=2)
    prob_model1 = SimulationDummy(mean=mu2, cov=dev2, ndim_y=2)

    gof1 = GoodnessOfFit(est, prob_model1, n_observations=10000)
    self.assertEqual(gof1.hellinger_distance(), np.sqrt(hd_squared_analytic_result))



if __name__ == '__main__':
  pytest.main('--html=unittest_report.html --self-contained-html')

