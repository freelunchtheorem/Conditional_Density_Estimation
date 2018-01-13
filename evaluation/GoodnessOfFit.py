from scipy.stats import chisquare, kstest, jarque_bera
from density_estimator import BaseDensityEstimator
import density_simulation.ProbabilisticModels as pm
from sklearn.model_selection import train_test_split




class GoodnessOfFit:
  def __init__(self, estimator, probabilistic_model, n_observations=1000, train_size=0.6):
    """

    :param estimator:
    :param probabilistic_model:
    :param n_observations:
    """
    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert type(probabilistic_model).__name__ in pm.get_probabilistic_models_list()


    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations


    self.X, self.Y = probabilistic_model.simulate(n_observations)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, random_state=42,
                                                                            train_size=train_size)

    self.proba_model_conditional_pdf = probabilistic_model.pdf(self.X, self.Y)
    self.proba_model_conditional_cdf = probabilistic_model.cdf(self.X, self.Y)

    estimator.fit(self.X_train, self.y_train, n_epoch=300)

    self.estimator = estimator
    self.estimator_cond_samples = self.estimator.sample(self.X_test)



  def chi_squared_test(self):
    chisq, p = chisquare(self.estimator_cond_samples)
    return chisq, p


  def kolmogorov_smirnov_test(self):
    ks, p = kstest(self.estimator_cond_samples, self.proba_model_conditional_cdf)
    return ks,p


  def jarque_bera_test(self):
    jb, p = jarque_bera(self.estimator_cond_samples)
    return jb, p


  def kl_divergence(self, P, Q):
    raise NotImplementedError