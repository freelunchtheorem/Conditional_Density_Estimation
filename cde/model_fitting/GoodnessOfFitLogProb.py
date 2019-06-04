import time
import numpy as np
import logging
import matplotlib.pyplot as plt
from ml_logger import logger

from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cde.density_simulation import BaseConditionalDensitySimulation
from cde.model_fitting.GoodnessOfFitSingleResult import GoodnessOfFitSingleResult


class GoodnessOfFitLogProb:
  """ Class that takes an estimator a probabilistic simulation model. The estimator is fitted on n_obervation samples.
  Then the goodness of fit w.r.t to the true probability distribution is evaluated

  Args:
    estimator: Estimator instance that implements the functionality from BaseDensityEstimator (can be either fitted or not fitted)
    probabilistic_model: ConditionalDensity instance which implements the methods simulate, pdf, cdf
    X_train: (ndarray) training data x
    Y_train: (ndarray) training data y
    X_test: (ndarray) test data x
    Y_test: (ndarray) training data x
    task_name: specifies a unique name fo the GoodnessOfFit run, e.g. KernelMixtureNetwork_task_19. If task_name was not set during call,
    the name was specified in the estimator object (estimator.name) is used. If it was not specified there either, it is set to
     estimator and prob_model name.

  """
  def __init__(self, estimator, probabilistic_model, X_train, Y_train, X_test, Y_test, task_name=None):

    assert isinstance(estimator, BaseDensityEstimator), "estimator must inherit BaseDensityEstimator class"
    assert isinstance(probabilistic_model, BaseConditionalDensitySimulation), "probabilistic model must inherit from ConditionalDensity"

    np.seterr(divide='ignore')

    self.probabilistic_model = probabilistic_model

    self.proba_model_conditional_pdf = probabilistic_model.pdf
    self.proba_model_conditional_cdf = probabilistic_model.cdf

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
    self.n_observations = X_train.shape[0]
    self.n_test_samples = Y_test.shape[0]

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
      self.estimator.fit(self.X_train, self.Y_train, verbose=False)
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
    assert self.estimator is not None
    assert self.probabilistic_model is not None

    gof_result = GoodnessOfFitSingleResult(self.estimator.get_configuration(), self.probabilistic_model.get_configuration())

    """ Evaluation stats """
    gof_result.n_observations = [self.n_observations]
    gof_result.time_to_fit = self.time_to_fit
    gof_result.score = self.estimator.score(self.X_test, self.Y_test)

    return gof_result

  def __str__(self):
    return str("{}\n{}\nGoodness of fit:\n n_observations: {}\n n_x_cond: {}".format(
      self.estimator, self.probabilistic_model, self.n_observations, self.x_cond))

