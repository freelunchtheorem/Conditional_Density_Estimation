import unittest
from density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork
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
    ks, p = gof.kolmogorov_smirnov_cdf()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("cdf-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)

  def test_gaussian_dummy_kolmogorov_2sample_1(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=2)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_2sample()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("2-sample-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)


  def test_gaussian_dummy_kolmogorov_cdf_2(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=3)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_cdf()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("cdf-based KS test (t, p): ", ks, p)
    self.assertLess(p, alpha)

  def test_gaussian_dummy_kolmogorov_2sample_2(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=3)
    gof = GoodnessOfFit(est, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_2sample()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("2-sample-based KS test (t, p): ", ks, p)
    self.assertLess(p, alpha)




  """ --- estimator checks with density estimators and simulators --- """

  def test_kmn_dummy_kolmogorov_2sample(self):
    est = KernelMixtureNetwork(train_scales=True, n_centers=50)
    prob_model = SimulationDummy(mean=1)
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    ks, p = gof.kolmogorov_smirnov_2sample()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("2-sample-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)

  def test_kmn_dummy_kolmogorov_cdf(self):
    est = KernelMixtureNetwork(train_scales=True, n_centers=50)
    prob_model = SimulationDummy(mean=1)
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    ks, p = gof.kolmogorov_smirnov_cdf()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("cdf-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)

  def test_kmn_econ_kolmogorov_cdf(self, alpha=0.05):
    est = KernelMixtureNetwork(train_scales=True, n_centers=50)
    prob_model = EconDensity()
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    ks_cdf, p_cdf = gof.kolmogorov_smirnov_cdf()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("cdf-based KS test (t, p): ", ks_cdf, p_cdf)
    self.assertGreater(p_cdf, alpha)

  def test_kmn_econ_kolmogorov_2sample(self, alpha=0.05):
    est = KernelMixtureNetwork(train_scales=True, n_centers=50)
    prob_model = EconDensity()
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp)
    self.assertGreater(p_2samp, alpha)

  def test_lscde_econ_kolmogorov_2sample(self, alpha=0.05):
    est = LSConditionalDensityEstimation(n_centers=50)
    prob_model = EconDensity()
    gof = GoodnessOfFit(est, prob_model, n_observations=4000)
    ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp)
    self.assertGreater(p_2samp, alpha)

  def test_gaussian_dummy_kl(self):
    est = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=7)
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertGreater(kl, 1)

  def test_kmn_econ_kl(self):
    est = KernelMixtureNetwork(train_scales=True, n_centers=100)
    prob_model = EconDensity()
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertLess(kl, 1)

  def test_kmn_gmm_kl_1(self):
    est = KernelMixtureNetwork(train_scales=True, n_centers=20)
    prob_model = GaussianMixture(n_kernels=5)
    gof = GoodnessOfFit(est, prob_model, n_observations=1000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertLess(kl, 1)

  def test_kmn_gmm_kl_2(self):
    est = KernelMixtureNetwork(train_scales=True, n_centers=20)
    prob_model = GaussianMixture(n_kernels=20, means_std=3)
    gof = GoodnessOfFit(est, prob_model, n_observations=1000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertLess(kl, 1)

  def test_lscde_econ_kl(self):
    est = LSConditionalDensityEstimation(n_centers=100, bandwidth=0.15, keep_edges=True)
    prob_model = EconDensity()
    gof = GoodnessOfFit(est, prob_model, n_observations=8000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertLess(kl, 1)

  def test_lscde_gmm_kl(self):
    est = LSConditionalDensityEstimation(n_centers=50)
    prob_model = GaussianMixture(n_kernels=20)
    gof = GoodnessOfFit(est, prob_model, n_observations=1000)
    kl = gof.kl_divergence()
    print("Estimator: ", est.__class__.__name__)
    print("Simulator: ", prob_model.__class__.__name__)
    print("KL-divergence: ", kl)
    self.assertLess(kl, 1)



if __name__ == '__main__':
  unittest.main()