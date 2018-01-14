from scipy.stats import shapiro, kstest, jarque_bera, ks_2samp
from density_estimator.base import BaseDensityEstimator
from sklearn.model_selection import train_test_split
from density_simulation import get_probabilistic_models_list
from density_simulation import ConditionalDensity
import numpy as np
import matplotlib.pyplot as plt



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


    self.estimator = estimator

    if X_cond is None:
      self.X_cond = np.asarray([0 for _ in range(n_x_cond)])

    _, self.estimator_cond_samples = self.estimator.sample(self.X_cond)
    _, self.proba_model_conditional_samples = probabilistic_model.simulate_conditional(self.X_cond)




  def shapiro_wilk_test(self):
    sw, p = shapiro(self.estimator_cond_samples)
    return sw, p

  def kolmogorov_smirnov_cdf(self):
    np.random.seed(98765431)
    ks, p = kstest(self.estimator_cond_samples, lambda y: self.probabilistic_model.cdf(self.X_cond, y))
    return ks, p

  def kolmogorov_smirnov_2sample(self):
    np.random.seed(98765431)
    ks, p = ks_2samp(self.estimator_cond_samples, self.proba_model_conditional_samples)
    return ks, p

  def jarque_bera_test(self):
    jb, p = jarque_bera(self.estimator_cond_samples)
    return jb, p

  def kl_divergence(self):
    Q = self.probabilistic_model.pdf
    P = self.estimator.predict

    # prepare mesh
    linspace_x = np.linspace(-5, 5, num=100)
    linspace_y = np.linspace(-5, 5, num=100)

    kl_divergence = 0

    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()

    Z_P = P(X,Y)
    Z_Q = Q(X,Y)
    xv, yv = np.meshgrid(linspace_x, linspace_y, sparse=False, indexing='ij')
    # for i in range(len(linspace_x)):
    #   for j in range(len(linspace_y)):
    #     X = xv[i,j]
    #     Y = yv[i,j]
    #     Z_P.append(P(X,Y))
    #      Z_Q.append(Q(X,Y))

    # todo: compute delta x
    kl = np.sum(np.where(Z_P != 0, Z_P * np.log(Z_P / Z_Q), 0))



    # treat xv[i,j], yv[i,j]



    # todo implement kl divergence with meshgrid
    raise NotImplementedError


