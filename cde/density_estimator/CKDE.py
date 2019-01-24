import numpy as np
import statsmodels.api as sm

from cde.utils.async_executor import execute_batch_async_pdf
from .BaseDensityEstimator import BaseDensityEstimator

MULTIPROC_THRESHOLD = 10**4

class ConditionalKernelDensityEstimation(BaseDensityEstimator):
  """ ConditionalKernelDensityEstimation (CKDE): Nonparametric conditional density estimator that
      models the joint probability p(x,y) and marginal probability p(x) via kernel density estimation
      and computes the conditional density as p(y|x) = p(x, y) / p(x). This implementation wraps
      functionality of the statsmodels.nonparametric module.

      Args:
          name: (str) name / identifier of estimator
          ndim_x: (int) dimensionality of x variable
          ndim_y: (int) dimensionality of y variable
          bandwidth: (array_like or str)
            If an array, it is a fixed user-specified bandwidth.  If a string,
            should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares
          n_jobs: (int) number of jobs to launch for calls with large batch sizes
          random_seed: (optional) seed (int) of the random number generators used

      References:
          Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
          Princeton University Press. (2007)
  """

  def __init__(self, name='CKDE', ndim_x=None, ndim_y=None, bandwidth='cv_ml', n_jobs=-1, random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)
    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.n_jobs = n_jobs
    self.random_seed = random_seed

    assert bandwidth in ['normal_reference', 'cv_ml', 'cv_ls']
    self.bandwidth = bandwidth

    self.fitted = False
    self.can_sample = False
    self.has_pdf = True
    self.has_cdf = True


  def fit(self, X, Y, **kwargs):
    """ Since CKDE is a lazy learner, fit just stores the provided training data (X,Y)

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
    self.y_mean, self.y_std = np.mean(Y, axis=0), np.std(Y, axis=0)

    dep_type = 'c' * self.ndim_y
    indep_type = 'c' * self.ndim_x
    self.sm_kde = sm.nonparametric.KDEMultivariateConditional(endog=[Y], exog=[X], dep_type=dep_type, indep_type=indep_type, bw=self.bandwidth)

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

    n_samples = X.shape[0]
    if n_samples >= MULTIPROC_THRESHOLD:
      return execute_batch_async_pdf(self.sm_kde.pdf, Y, X, n_jobs=self.n_jobs)
    else:
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
    n_samples = X.shape[0]
    if n_samples > MULTIPROC_THRESHOLD:
      execute_batch_async_pdf(self.sm_kde.cdf, Y, X, n_jobs=self.n_jobs)
    else:
      return self.sm_kde.cdf(endog_predict=Y, exog_predict=X)

  def sample(self, X):
    raise NotImplementedError("Conditional Kernel Density Estimation is a lazy learner and does not support sampling")

  def _param_grid(self):
    mean_std_y = np.mean(self.y_std)
    bandwidths = np.asarray([0.01, 0.1, 0.5, 1, 2, 5]) * mean_std_y

    param_grid = {
      "bandwidth": bandwidths
    }
    return param_grid


  def __str__(self):
    return "\n Estimator type: {}\n ndim_x: {}\n ndim_y: {}\n bandwidth: {}\n".format(self.__class__.__name__, self.ndim_x, self.ndim_y,
                                                                                             self.bandwidth)

  def __unicode__(self):
    return self.__str__()

