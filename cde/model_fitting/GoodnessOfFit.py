import time
import numpy as np
import logging
import matplotlib.pyplot as plt
from ml_logger import logger

from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cde.density_simulation import BaseConditionalDensitySimulation
from cde.model_fitting.GoodnessOfFitSingleResult import GoodnessOfFitSingleResult
from cde.model_fitting.divergences import divergence_measures_pdf


class GoodnessOfFit:
  """ Class that takes an estimator a probabilistic simulation model. The estimator is fitted on n_obervation samples.
  Then the goodness of fit w.r.t to the true probability distribution is evaluated

  Args:
    estimator: Estimator instance that implements the functionality from BaseDensityEstimator (can be either fitted or not fitted)
    probabilistic_model: ConditionalDensity instance which implements the methods simulate, pdf, cdf
    n_observations: number of observations which are drawn from the probabilistic_model to fit the estimator.
                    If the estimator is already fitted, this parameter is ignored
    n_x_cond: number of different x to condition on - e.g. if n_x_cond=20, p(y|x) is evaluated for 20 different x
    seed: random seed to draw samples from the probabilistic model
    task_name: specifies a unique name fo the GoodnessOfFit run, e.g. KernelMixtureNetwork_task_19. If task_name was not set during call,
    the name was specified in the estimator object (estimator.name) is used. If it was not specified there either, it is set to
     estimator and prob_model name.

  """
  def __init__(self, estimator, probabilistic_model, X, Y, n_observations, x_cond, n_mc_samples, seed=24, task_name=None,
               tail_measures=True):

    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, BaseConditionalDensitySimulation), "probabilistic model must inherit from ConditionalDensity"

    np.seterr(divide='ignore')

    self.probabilistic_model = probabilistic_model
    self.n_observations = n_observations
    self.x_cond = x_cond
    self.n_mc_samples = n_mc_samples

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    self.seed = seed
    np.random.seed(seed)
    self.X = X
    self.Y = Y
    self.n_observations = n_observations
    self.tail_measures = tail_measures

    self.estimator = estimator

    if task_name is not None:
      self.task_name = task_name
    elif hasattr(self.estimator, 'name'):
      self.task_name = str(self.estimator.name)
    else:
      self.task_name = type(self.estimator).__name__ + '_' + type(self.probabilistic_model).__name__

  def fit_estimator(self, print_fit_result=True): #todo set to False
    """
    Fits the estimator with the provided data

    Args:
      print_fit_result: boolean that specifies whether the fitted distribution shall be plotted (only works if ndim_x and ndim_y = 1)
    """

    self.time_to_fit = None
    if not self.estimator.fitted:  # fit estimator if necessary
      t_start = time.time()
      self.estimator.fit(self.X, self.Y, verbose=False)
      self.time_to_fit = (time.time() - t_start) * self.n_observations / 1000  # time to fit per 1000 samples

    if print_fit_result and self.estimator.fitted:
      if self.probabilistic_model.ndim_x == 1 and self.probabilistic_model.ndim_y == 1:
        plt3d_true = self.probabilistic_model.plot(mode="pdf", numpyfig=False)
        logger.log_pyplot(key=self.task_name, fig=plt3d_true)
        plt.close(plt3d_true)

      if self.estimator.ndim_x == 1 and self.estimator.ndim_y == 1:
        plt2d = self.estimator.plot2d(show=False, numpyfig=False)
        plt3d = self.estimator.plot3d(show=False, numpyfig=False)
        logger.log_pyplot(key=self.task_name + "_fitted_cond_distr_2d", fig=plt2d)
        logger.log_pyplot(key=self.task_name + "_fitted_cond_distr_3d", fig=plt3d)
        plt.close(plt2d)
        plt.close(plt3d)

  def compute_results(self):
    """
      Computes statistics and stores the results in GoodnessOfFitResult object

      Returns:
        GoodnessOfFitResult object that holds the computed statistics
    """
    assert self.x_cond.all()
    assert self.estimator is not None
    assert self.probabilistic_model is not None

    gof_result = GoodnessOfFitSingleResult(self.estimator.get_configuration(), self.probabilistic_model.get_configuration(),
                                           x_cond=self.x_cond)

    if self.n_mc_samples < 10**5:
      logging.warning("using less than 10**5 samples for monte carlo not recommended")

    """ Evaluation stats """
    gof_result.n_observations = [self.n_observations]
    gof_result.x_cond = [str(self.x_cond.flatten())]
    gof_result.x_cond_ = self.x_cond # original data preserved
    gof_result.n_mc_samples = [self.n_mc_samples]

    """ Divergence measures """
    gof_result.hellinger_distance_, gof_result.kl_divergence_, gof_result.js_divergence_ =\
       divergence_measures_pdf(self.probabilistic_model, self.estimator, x_cond=self.x_cond, n_samples=self.n_mc_samples)

    gof_result.hellinger_distance = [np.mean(gof_result.hellinger_distance_)]
    gof_result.kl_divergence = [np.mean(gof_result.kl_divergence_)]
    gof_result.js_divergence = [np.mean(gof_result.js_divergence_)]

    """ Mean and Std """
    """ create strings since pandas requires lists to be all of the same length if numerical """
    # estimator
    gof_result.mean_est_ = self.estimator.mean_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
    gof_result.std_est_ = self.estimator.std_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
    gof_result.mean_est, gof_result.std_est = [str(gof_result.mean_est_.flatten())], [str(gof_result.std_est_.flatten())]

    # simulator
    gof_result.mean_sim_ = self.probabilistic_model.mean_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
    gof_result.std_sim_ = self.probabilistic_model.std_(self.x_cond, n_samples=self.n_mc_samples) # original data preserved
    gof_result.mean_sim, gof_result.std_sim = [str(gof_result.mean_sim_ .flatten())], [str(gof_result.std_sim_.flatten())]

    # absolute mean, std difference
    gof_result.mean_abs_diff = np.mean(np.abs(gof_result.mean_est_ - gof_result.mean_sim_))
    gof_result.std_abs_diff = np.mean(np.abs(gof_result.std_sim_ - gof_result.std_sim_))

    """ tail risk """
    if self.estimator.ndim_y == 1 and self.tail_measures:
      # estimator
      gof_result.VaR_est_, gof_result.CVaR_est_ = self.estimator.tail_risk_measures(self.x_cond, n_samples=self.n_mc_samples)
      gof_result.VaR_est, gof_result.CVaR_est = [str(gof_result.VaR_est_.flatten())], [str(gof_result.CVaR_est_.flatten())]

      # simulator
      gof_result.VaR_sim_, gof_result.CVaR_sim_ = self.probabilistic_model.tail_risk_measures(self.x_cond, n_samples=self.n_mc_samples)
      gof_result.VaR_sim, gof_result.CVaR_sim = [str(gof_result.VaR_sim_.flatten())], [str(gof_result.CVaR_sim_.flatten())]

      gof_result.VaR_abs_diff = np.mean(np.abs(gof_result.VaR_sim_ - gof_result.VaR_est_))
      gof_result.CVaR_abs_diff = np.mean(np.abs(gof_result.CVaR_sim_ - gof_result.CVaR_est_))

    """ time to fit """
    gof_result.time_to_fit = self.time_to_fit

    return gof_result

  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}".format(
      self.estimator, self.probabilistic_model, self.n_observations, self.x_cond))


def sample_x_cond(X, n_x_cond=20, low_percentile = 10, high_percentile=90, random_seed=92):
  """
  uniformly samples n_xcond points within the specified percentiles in X

  Args:
    X: data on which the percentiles shall be computed - ndarray with shape (n_samples, ndim_x)
    n_x_cond: number of x_cond points to be sampled
    low_percentile: lower percentile (int)
    high_percentile: upper percentile (int)

  Returns:
    sampled x_cond points - ndarray of shape (n_xcond, ndim_x)
  """
  assert 0 <= low_percentile < high_percentile <= 100
  rand = np.random.RandomState(random_seed)

  if X.ndim == 1:
    X = np.expand_dims(X, axis=1)

  samples_per_dim = []
  for i in range(X.shape[1]):
    low = np.percentile(X[:,i], low_percentile)
    high = np.percentile(X[:,i], high_percentile)
    samples_per_dim.append(rand.uniform(low, high, size=(n_x_cond, 1)))

  x_cond = np.hstack(samples_per_dim)

  assert x_cond.shape[1] == X.shape[1] and x_cond.shape[0] == n_x_cond
  return x_cond
