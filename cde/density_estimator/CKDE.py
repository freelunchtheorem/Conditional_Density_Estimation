import numpy as np
from .BaseDensityEstimator import BaseDensityEstimator

import statsmodels.api as sm


class ConditionalKernelDensityEstimation(BaseDensityEstimator):


  def __init__(self, name, ndim_x, ndim_y, bandwidth_selection='cv_ml', random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)

    assert bandwidth_selection in ['normal_reference', 'cv_ml', 'cv_ls']
    self.bandwidth_selection = bandwidth_selection

    self.fitted = False
    self.can_sample = False
    self.has_pdf = True
    self.has_cdf = True

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y

  def fit(self, X, Y, **kwargs):
    """ Since CKDE is a lazy learner, fit just stores the provided training data (X,Y)

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    dep_type = 'c' * self.ndim_y
    indep_type = 'c' * self.ndim_x
    self.sm_kde = sm.nonparametric.KDEMultivariateConditional(endog=[Y], exog=[X], dep_type=dep_type, indep_type=indep_type, bw=self.bandwidth_selection)

    self.fitted = True
    self.can_sample = False
    self.has_cdf = True

  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """
    X,Y = self._handle_input_dimensionality(X, Y)
    return self.sm_kde.pdf(endog_predict=Y, exog_predict=X)

  def cdf(self, X, Y):
    """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

    Args:
      X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_samples, n_dim_y)

    Returns:
      conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )

    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.sm_kde.cdf(endog_predict=Y, exog_predict=X)

  def sample(self, X):
    raise NotImplementedError("Conditional Kernel Density Estimation is a lazy learner and does not support sampling")

  def _param_grid(self):
    mean_std_y = np.mean(self.y_std)
    mean_std_x = np.mean(self.x_std)
    bandwidths = np.asarray([0.01, 0.1, 0.5, 1, 2, 5]) * mean_std_y
    epsilons = np.asarray([0.001, 0.1, 0.5, 1]) * mean_std_x

    param_grid = {
      "bandwidth": bandwidths,
      "epsilon": epsilons,
      "weighted": [True, False]
    }
    return param_grid


  def __str__(self):
    return "\nEstimator type: {}\n  epsilon: {}\n weighted: {}\n bandwidth: {}\n".format(self.__class__.__name__, self.epsilon, self.weighted,
                                                                                             self.bandwidth)

  def __unicode__(self):
    return self.__str__()

