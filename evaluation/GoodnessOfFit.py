from scipy.stats import shapiro, kstest, jarque_bera, ks_2samp
from density_estimator.base import BaseDensityEstimator
from sklearn.model_selection import train_test_split
from density_simulation import get_probabilistic_models_list
from density_simulation import ConditionalDensity
import numpy as np
import scipy


class GoodnessOfFit:
  def __init__(self, estimator, probabilistic_model, X_cond = None, n_observations=10000, n_x_cond=10000, print_fit_result=False):
    """

    :param estimator:
    :param probabilistic_model:
    :param n_observations:
    """
    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, ConditionalDensity), "probabilistic model must inherit from ConditionalDensity"


    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations


    self.X, self.Y = probabilistic_model.simulate(self.n_observations)

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    if not estimator.fitted:
      estimator.fit(self.X, self.Y)

    if print_fit_result:
      estimator.plot_loss()
      self.probabilistic_model.plot(mode="cdf")
      self.probabilistic_model.plot(mode="pdf")


    self.estimator = estimator

    if X_cond is None:
      """ generate X_cond data with shape (n_x_cond, estimator.ndim_x) """
      X_cond = np.stack([np.asarray([0 for _ in range(n_x_cond)]) for i in range(self.estimator.ndim_x)], axis=1)


    self.X_cond = X_cond

    self.resample_new_conditional_samples()


  def resample_new_conditional_samples(self):
    _, self.estimator_conditional_samples = self.estimator.sample(self.X_cond)
    _, self.proba_model_conditional_samples = self.probabilistic_model.simulate_conditional(self.X_cond)

    """ kstest can't handle single-dimensional entries, therefore remove it"""
    if self.estimator_conditional_samples.ndim == 2:
      if self.estimator_conditional_samples.shape[1] == 1:
        self.estimator_conditional_samples = np.squeeze(self.estimator_conditional_samples, axis = 1)
    if self.proba_model_conditional_samples.ndim == 2:
      if self.proba_model_conditional_samples.shape[1] == 1:
        self.proba_model_conditional_samples = np.squeeze(self.proba_model_conditional_samples, axis=1)


  def shapiro_wilk_test(self):
    sw, p = shapiro(self.estimator_conditional_samples)
    return sw, p

  def kolmogorov_smirnov_cdf(self, repeat=20):
    np.random.seed(98765431)
    ks = []
    p = []
    for _ in range(repeat):
      self.resample_new_conditional_samples()
      ks_new, p_new = kstest(self.estimator_conditional_samples, lambda y: self.probabilistic_model.cdf(self.X_cond, y))
      ks.append(ks_new), p.append(p_new)
    return np.mean(ks), np.mean(p)

  def kolmogorov_smirnov_2sample(self, repeat=20):
    np.random.seed(98765431)
    ks = []
    p = []
    for _ in range(repeat):
      self.resample_new_conditional_samples()
      ks_new, p_new = ks_2samp(self.estimator_conditional_samples, self.proba_model_conditional_samples)
      ks.append(ks_new), p.append(p_new)
    return np.mean(ks), np.mean(p)

  def jarque_bera_test(self):
    jb, p = jarque_bera(self.estimator_conditional_samples)
    return jb, p

  def kl_divergence(self):
    P = self.probabilistic_model.pdf
    Q = self.estimator.predict_density

    # prepare mesh
    linspace_x = np.linspace(-5, 5, num=100)
    linspace_y = np.linspace(-5, 5, num=100)

    xv, yv = np.meshgrid(linspace_x, linspace_y, sparse=False, indexing='xy')
    X, Y = xv.flatten(), yv.flatten()

    Z_P = P(X,Y)
    Z_Q = Q(X,Y)

    # KL can't handle zero values -> replace with small values if zero values existent
    # Z_Q[Z_Q == 0] = np.finfo(np.double).tiny
    # Z_P[Z_P == 0] = np.finfo(np.double).tiny

    return scipy.stats.entropy(pk=Z_P, qk=Z_Q)






