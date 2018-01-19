import unittest
import pytest
from density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, NeighborKernelDensityEstimation
from density_simulation import EconDensity, GaussianMixture
from evaluation.GoodnessOfFit import GoodnessOfFit
from tests.Dummies import GaussianDummy, SimulationDummy

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
    self.assertLess(gof_results.mean_kl, 0.01)


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
    print("KL-Divergence:", gof_results.mean_kl)
    print(gof)
    self.assertGreater(gof_results.mean_kl, 0.01)

  def test_gaussian_dummy_kl_divergence2(self):
    est = GaussianDummy(mean=2)
    prob_model1 = SimulationDummy(mean=4)
    gof1 = GoodnessOfFit(est, prob_model1, n_observations=10000)
    gof_results1 = gof1.compute_results()

    prob_model2 = SimulationDummy(mean=3)
    gof2 = GoodnessOfFit(est, prob_model2, n_observations=10000)
    gof_results2 = gof2.compute_results()
    self.assertLess(gof_results2.mean_kl, gof_results1.mean_kl)

  #
  #
  # """ --- estimator checks with density estimators and simulators --- """
  #
  # def test_kmn_dummy_kolmogorov_2sample(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=50)
  #   prob_model = SimulationDummy(mean=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   ks, p = gof.kolmogorov_smirnov_2sample()
  #   print("2-sample-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_kmn_dummy_kolmogorov_cdf(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=50)
  #   prob_model = SimulationDummy(mean=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   ks, p = gof.kolmogorov_smirnov_cdf()
  #   print("cdf-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_kmn_econ_kolmogorov_cdf(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=60)
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   ks_cdf, p_cdf = gof.kolmogorov_smirnov_cdf()
  #   print("cdf-based KS test (t, p): ", ks_cdf, p_cdf, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p_cdf, alpha)
  #
  # def test_kmn_econ_kolmogorov_2sample(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=60, n_training_epochs=600)
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
  #   print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p_2samp, alpha)
  #
  # def test_lscde_econ_kolmogorov_2sample(self):
  #   est = LSConditionalDensityEstimation(n_centers=50)
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=4000)
  #   ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
  #   print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p_2samp, alpha)
  #
  # def test_gaussian_dummy_kl(self):
  #   est = GaussianDummy(mean=2)
  #   prob_model = SimulationDummy(mean=7)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence (pass if < 1)", kl)
  #   print(gof)
  #   self.assertGreater(kl, 1)
  #
  # def test_kmn_econ_kl(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=60, n_training_epochs=600)
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=8000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence (pass if < 1)", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_kmn_gmm_kl_1(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=20)
  #   prob_model = GaussianMixture(n_kernels=5)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence (pass if < 1)", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_kmn_gmm_kl_2(self):
  #   est = KernelMixtureNetwork(train_scales=True, n_centers=20)
  #   prob_model = GaussianMixture(n_kernels=20, means_std=3)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence (pass if < 1)", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_lscde_econ_kl(self):
  #   est = LSConditionalDensityEstimation(n_centers=50, bandwidth=0.15, keep_edges=True)
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence: ", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_lscde_gmm_kl(self):
  #   est = LSConditionalDensityEstimation(n_centers=50)
  #   prob_model = GaussianMixture(n_kernels=20)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence (pass if < 1)", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_nkde_dummy_kolmogorov_cdf(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = SimulationDummy(mean=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_cdf()
  #   print("cdf-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_dummy_kolmogorov_2sample(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = SimulationDummy(mean=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_2sample()
  #   print("2-sample-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_dummy_kl(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = SimulationDummy(mean=1)
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence: ", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_nkde_econ_kolmogorov_cdf(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_cdf()
  #   print("2-sample-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_econ_kolmogorov_2sample(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_2sample()
  #   print("cdf-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_econ_kl(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = EconDensity()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence: ", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  # def test_nkde_gmm_cdf(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = GaussianMixture()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_cdf()
  #   print("cdf-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_gmm_kolmogorov_2sample(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = GaussianMixture()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   ks, p = gof.kolmogorov_smirnov_2sample()
  #   print("cdf-based KS test (t, p): ", ks, p, "alpha: ", alpha)
  #   print(gof)
  #   self.assertGreater(p, alpha)
  #
  # def test_nkde_gmm_kl(self):
  #   est = NeighborKernelDensityEstimation()
  #   prob_model = GaussianMixture()
  #   gof = GoodnessOfFit(est, prob_model, n_observations=1000)
  #   kl = gof.kl_divergence()
  #   print("KL-divergence: ", kl)
  #   print(gof)
  #   self.assertLess(kl, 1)
  #
  #


if __name__ == '__main__':
  pytest.main('--html=report.html --self-contained-html')

