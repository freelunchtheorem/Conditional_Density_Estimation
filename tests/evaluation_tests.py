import unittest
from density_estimator import LSConditionalDensityEstimation, NeighborKernelDensityEstimation, KernelMixtureNetwork
from density_simulation import EconDensity, GMM
from evaluation.GoodnessOfFit import GoodnessOfFit
from tests.Dummies import GaussianDummy, SimulationDummy



class EvaluationTest(unittest.TestCase):

  """ --- plausibility checks with dummy (gaussian) estimators and simulators --- """

  def test_gaussian_kolmogorov_cdf_1(self, alpha=0.05):
    gd = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=2)
    gof = GoodnessOfFit(gd, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_cdf()
    print("cdf-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)

  def test_gaussian_kolmogorov_2sample_1(self, alpha=0.05):
    gd = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=2)
    gof = GoodnessOfFit(gd, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_2sample()
    print("2-sample-based KS test (t, p): ", ks, p)
    self.assertGreater(p, alpha)


  def test_gaussian_kolmogorov_cdf_2(self, alpha=0.05):
    gd = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=3)
    gof = GoodnessOfFit(gd, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_cdf()
    print("cdf-based KS test (t, p): ", ks, p)
    self.assertLess(p, alpha)

  def test_gaussian_kolmogorov_2sample_2(self, alpha=0.05):
    gd = GaussianDummy(mean=2)
    prob_model = SimulationDummy(mean=3)
    gof = GoodnessOfFit(gd, prob_model, n_observations=10000)
    ks, p = gof.kolmogorov_smirnov_2sample()
    print("2-sample-based KS test (t, p): ", ks, p)
    self.assertLess(p, alpha)


  """ --- estimator checks with real estimators and simulators --- """

  def test_kmn_econ_kolmogorov_cdf(self, alpha=0.05):
    kmn = KernelMixtureNetwork(train_scales=True, n_centers=30)
    prob_model = EconDensity()
    gof = GoodnessOfFit(kmn, prob_model, n_observations=10000)

    ks_cdf, p_cdf = gof.kolmogorov_smirnov_cdf()
    print("cdf-based KS test (t, p): ", ks_cdf, p_cdf)
    self.assertGreater(p_cdf, alpha)

  def test_kmn_econ_kolmogorov_2sample(self, alpha=0.05):
    kmn = KernelMixtureNetwork(train_scales=True, n_centers=30)
    prob_model = EconDensity()
    gof = GoodnessOfFit(kmn, prob_model, n_observations=10000)

    ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
    print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp)
    self.assertGreater(p_2samp, alpha)


  def test_lscde_econ_kolmogorov_2sample(self, alpha=0.05):
    lscde = LSConditionalDensityEstimation()
    prob_model = EconDensity()
    gof = GoodnessOfFit(lscde, prob_model, n_observations=5000)

    ks_2samp, p_2samp = gof.kolmogorov_smirnov_2sample()
    print("2-sample-based KS test (t, p): ", ks_2samp, p_2samp)
    self.assertGreater(p_2samp, alpha)




if __name__ == '__main__':
  unittest.main()