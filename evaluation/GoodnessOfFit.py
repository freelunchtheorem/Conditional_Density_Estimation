from scipy.stats import shapiro, kstest, jarque_bera, ks_2samp
from density_estimator.base import BaseDensityEstimator
from density_simulation import ConditionalDensity
from joblib import Parallel, delayed
import numpy as np
import scipy


class GoodnessOfFit:
  def __init__(self, estimator, probabilistic_model, X_cond = None, n_observations=100, n_x_cond=1000, print_fit_result=False,
               repeat_kolmogorov=3):
    """

    :param estimator:
    :param probabilistic_model:
    :param n_observations:
    """
    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, ConditionalDensity), "probabilistic model must inherit from ConditionalDensity"


    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations

    self.n_x_cond = n_x_cond
    self.repeat_kolmogorov = repeat_kolmogorov

    self.X, self.Y = probabilistic_model.simulate(self.n_observations)

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    if not estimator.fitted:
      estimator.fit(self.X, self.Y)

    if print_fit_result:
      self.probabilistic_model.plot(mode="pdf")
      estimator.plot()

    self.estimator = estimator

    if X_cond is None:
      """ generate X_cond data with shape (n_x_cond, estimator.ndim_x) """
      X_cond = np.stack([np.asarray([0 for _ in range(self.n_x_cond)]) for i in range(self.estimator.ndim_x)], axis=1)
      """ in the case X_Cond is (n_x_cond, 1) convert to (n_x_cond, ) """
      X_cond = np.squeeze(X_cond)

    self.X_cond = X_cond
    # set initial values
    self.estimator_conditional_samples, self.proba_model_conditional_samples = self.sample_conditional_values()


  def sample_conditional_values(self):
    _, estimator_conditional_samples = self.estimator.sample(self.X_cond)
    _, proba_model_conditional_samples = self.probabilistic_model.simulate_conditional(self.X_cond)

    """ kstest can't handle single-dimensional entries, therefore remove it"""
    if estimator_conditional_samples.ndim == 2:
      if estimator_conditional_samples.shape[1] == 1:
        estimator_conditional_samples = np.squeeze(estimator_conditional_samples, axis = 1)
    if proba_model_conditional_samples.ndim == 2:
      if proba_model_conditional_samples.shape[1] == 1:
        proba_model_conditional_samples = np.squeeze(proba_model_conditional_samples, axis=1)

    return estimator_conditional_samples, proba_model_conditional_samples

  def shapiro_wilk_test(self):
    sw, p = shapiro(self.estimator_conditional_samples)
    return sw, p

  def jarque_bera_test(self):
    jb, p = jarque_bera(self.estimator_conditional_samples)
    return jb, p

  def kolmogorov_smirnov_cdf(self):
    samples = [self.sample_conditional_values() for _ in range(self.repeat_kolmogorov)]
    estimator_cond_samples = np.asarray(samples)[:, 0]

    cdf = self.probabilistic_model.cdf
    X_cond = self.X_cond

    statistics = np.asarray(Parallel(n_jobs=-1)(delayed(ktest_cdf)(cdf, X_cond, estimator_cond_samples[i]) for i in range(
      self.repeat_kolmogorov)))
    return np.mean(statistics[:,0]), np.mean(statistics[:,1])

  def kolmogorov_smirnov_2sample(self):
    samples = np.asarray([self.sample_conditional_values() for _ in range(self.repeat_kolmogorov)])
    estimator_cond_samples = samples[:, 0]
    probabilistic_cond_samples = samples[:, 1]

    statistics = np.asarray(Parallel(n_jobs=-1)(delayed(ktest_2sample)(estimator_cond_samples[i], probabilistic_cond_samples[i]) for i in range(
      self.repeat_kolmogorov)))
    return np.mean(statistics[:,0]), np.mean(statistics[:,1])

  def kl_divergence(self):
    P = self.probabilistic_model.pdf
    Q = self.estimator.predict

    # prepare mesh
    linspace_x = np.linspace(-5, 5, num=100)
    linspace_y = np.linspace(-5, 5, num=100)

    xv, yv = np.meshgrid(linspace_x, linspace_y, sparse=False, indexing='xy')
    X, Y = xv.flatten(), yv.flatten()

    Z_P = P(X,Y)
    Z_Q = Q(X,Y)

    return scipy.stats.entropy(pk=Z_P, qk=Z_Q)

  def compute_results(self):
    x_cond = get_variable_grid(self.X, resolution=20)

    gof_result = GoodnessOfFitResults(x_cond)

    for i in range(x_cond.shape[0]):
      self.X_cond = np.vstack([x_cond[i, :] for _ in range(self.n_x_cond)])
      assert self.X_cond.shape == (self.n_x_cond, self.estimator.ndim_x)
      gof_result.ks_stat[i], gof_result.ks_pval[i] = self.kolmogorov_smirnov_cdf()

    gof_result.mean_kl = self.kl_divergence()
    gof_result.compute_means()
    return gof_result


  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}\n repeat_kolmogorov: {}\n".format(self.estimator,
                                                                                                                 self.probabilistic_model,
                                                                                                                 self.n_observations,

                                                                                                                 self.n_x_cond, self.repeat_kolmogorov))



class GoodnessOfFitResults:

  def __init__(self, x_cond):
    self.cond_values = x_cond

    self.kl = np.zeros(x_cond.shape[0])
    self.ks_stat = np.zeros(x_cond.shape[0])
    self.ks_pval = np.zeros(x_cond.shape[0])

    self.mean_kl = None
    self.mean_ks_stat = None
    self.mean_ks_pval = None

  def compute_means(self):
    self.mean_ks_stat = self.ks_stat.mean()
    self.mean_ks_pval = self.ks_pval.mean()


  def __str__(self):
    return "KL-Divergence: %.4f , KS Stat: %.4f, KS pval: %.4f"%(self.mean_kl, self.mean_ks_stat, self.mean_ks_pval)




""" closured functions cannot be pickled -> required to be outside for parallel computing """
def ktest_cdf(cdf, X_cond, est_cond_samples):
  return kstest(est_cond_samples, lambda y: cdf(X_cond, y))


def ktest_2sample(estimator_cond_samples, probabilistic_cond_samples):
  return scipy.stats.ks_2samp(estimator_cond_samples, probabilistic_cond_samples)

def get_variable_grid(X, resolution=20, low_percentile = 10, high_percentile=90):
  if X.ndim == 1:
    X = np.expand_dims(X, axis=1)

  linspaces = []
  for i in range(X.shape[1]):
    low = np.percentile(X[:,i], low_percentile)
    high = np.percentile(X[:,i], high_percentile)
    linspaces.append(np.linspace(low,high, num=resolution))

  grid = np.vstack([grid_dim.flatten() for grid_dim in np.meshgrid(*linspaces)]).T

  assert grid.shape[1] == X.shape[1]
  return grid