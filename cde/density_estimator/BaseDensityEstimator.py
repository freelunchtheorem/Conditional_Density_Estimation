from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.model_selection import cross_validate

from cde import ConditionalDensity

from cde.utils.center_point_select import *

class BaseDensityEstimator(ConditionalDensity):
  """ Interface for conditional density estimation models """

  def fit(self, X, Y, verbose=False):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
    """
    raise NotImplementedError

  def eval_by_cv(self, X, Y, n_splits=5, verbose=True):
    """ Fits the conditional density model with cross-validation by using the score function of the BaseDensityEstimator for
    scoring the various splits.

    Args:
      X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_samples, n_dim_y)
      n_splits: number of cross-validation folds (positive integer)
      verbose: the verbosity level
    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
    cv_results = cross_validate(self, X=X, y=Y, cv=n_splits, return_estimator=True, verbose=verbose)

    test_scores = cv_results['test_score']
    test_scores_max_idx = np.nanargmax(test_scores)
    estimator = cv_results['estimator'][test_scores_max_idx]

    self.set_params(**estimator.get_params())
    self.fit(X, Y)

  def fit_by_cv(self, X, Y, n_folds=3, param_grid=None, verbose=True, n_jobs=-1, random_state=None):
    """ Fits the conditional density model with hyperparameter search and cross-validation.
    - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
      the conditional log-likelihood is used for simulation_eval.
    - Fits the model with the previously selected hyperparameter configuration
    Args:
      X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_samples, n_dim_y)
      n_folds: number of cross-validation folds (positive integer)
      param_grid: (optional) a dictionary with the hyperparameters of the model as key and and a list of respective \
                  parametrizations as value. The hyperparameter search is performed over the cartesian product of \
                  the provided lists.
                  Example:
                  {"n_centers": [20, 50, 100, 200],
                   "center_sampling_method": ["agglomerative", "k_means", "random"],
                   "keep_edges": [True, False]
                  }
    """

    # save properties of data
    self.n_samples = X.shape[0]
    self.x_std = np.std(X, axis=0)
    self.y_std = np.std(Y, axis=0)

    if param_grid is None:
      param_grid = self._param_grid()

    cv_model = GridSearchCV(self, param_grid, n_jobs=n_jobs, refit=True, cv=n_folds, verbose=verbose, )
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # don't print division by zero warning
      cv_model.fit(X, Y)
    best_params = cv_model.best_params_
    if verbose: print("Cross-Validation terminated")
    if verbose: print("Best likelihood score: %.4f" % cv_model.best_score_)
    if verbose: print("Best params:", best_params)
    self.set_params(**best_params)
    self.fit(X, Y)
    return best_params

  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """
    raise NotImplementedError

  def log_pdf(self, X, Y):
    """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional log-probability log p(y|x) - numpy array of shape (n_query_samples, )

     """
    # This method is numerically unfavorable and should be overwritten with a numerically stable method
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      log_prob = np.log(self.pdf(X, Y))
    return log_prob

  def _param_grid(self):
    raise NotImplementedError

  def score(self, X, Y):
    """Computes the mean conditional log-likelihood of the provided data (X, Y)

    Args:
      X: numpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_query_samples, n_dim_y)

    Returns:
      average log likelihood of data
    """
    return np.mean(self.log_pdf(X, Y))

  def mean_(self, x_cond, n_samples=10**6):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2

    if self.has_pdf:
      return self._mean_pdf(x_cond, n_samples=n_samples)
    else:
      return self._mean_mc(x_cond, n_samples=n_samples)

  def std_(self, x_cond, n_samples=10 ** 6):
    """ Standard deviation of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Standard deviations  sqrt(Var[y|x]) corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    return self._std_pdf(x_cond, n_samples=n_samples)

  def covariance(self, x_cond, n_samples=10**6):
    """ Covariance of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    return self._covariance_pdf(x_cond, n_samples=n_samples)

  def skewness(self, x_cond, n_samples=10**6):
    """ Skewness of the fitted distribution conditioned on x_cond

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

       Returns:
         Skewness Skew[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
       """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    return self._skewness_pdf(x_cond, n_samples=n_samples)

  def kurtosis(self, x_cond, n_samples=10**6):
    """ Kurtosis of the fitted distribution conditioned on x_cond

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

       Returns:
         Kurtosis Kurt[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
       """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    return self._kurtosis_pdf(x_cond, n_samples=n_samples)

  def mean_std(self, x_cond, n_samples=10 ** 6):
    """ Computes Mean and Covariance of the fitted distribution conditioned on x_cond.
        Computationally more efficient than calling mean and covariance computatio separately

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] and Covariances Cov[y|x]
    """
    mean = self.mean_(x_cond, n_samples=n_samples)
    std = self._std_pdf(x_cond, n_samples=n_samples, mean=mean)
    return mean, std

  def value_at_risk(self, x_cond, alpha=0.01, n_samples=10**6):
    """ Computes the Value-at-Risk (VaR) of the fitted distribution. Only if ndim_y = 1

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      alpha: quantile percentage of the distribution

    Returns:
       VaR values for each x to condition on - numpy array of shape (n_values)
    """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    if self.has_cdf:
      VaR =  self._quantile_cdf(x_cond, alpha=alpha)
      if np.isnan(VaR).any() and self.can_sample: # try with sampling if failed
        VaR = self._quantile_mc(x_cond, alpha=alpha, n_samples=n_samples)
    elif self.can_sample:
      VaR =  self._quantile_mc(x_cond, alpha=alpha, n_samples=n_samples)
    else:
      raise NotImplementedError()
    return VaR

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**6):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of the fitted distribution. Only if ndim_y = 1

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
         alpha: quantile percentage of the distribution

       Returns:
         CVaR values for each x to condition on - numpy array of shape (n_values)
       """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)

    if self.has_pdf:
      return self._conditional_value_at_risk_mc_pdf(VaRs, x_cond, alpha=alpha, n_samples=n_samples)
    elif self.can_sample:
      return self._conditional_value_at_risk_sampling(VaRs, x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError("Distribution object must either support pdf or sampling in order to compute CVaR")

  def get_configuration(self, deep=True):
    """ Get parameter configuration for this estimator.

    Args:
      deep: boolean, optional If True, will return the parameters for this estimator and \
             contained subobjects that are estimators.

    Returns:
      params - mapping of string to any Parameter names mapped to their values.

    """
    param_dict = super(BaseDensityEstimator, self).get_params(deep=deep)
    param_dict['estimator'] = self.__class__.__name__

    for x in ["n_centers", "center_sampling_method", "x_noise_std", "y_noise_std",
              "random_seed", "ndim_x", "ndim_y"]:
      if hasattr(self, x):
        param_dict[x] = getattr(self, x)
      else:
        param_dict[x] = None

    return param_dict

  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 6):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo model_fitting

        Returns:
          - VaR values for each x to condition on - numpy array of shape (n_values)
          - CVaR values for each x to condition on - numpy array of shape (n_values)
        """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)

    if self.has_pdf:
      CVaRs = self._conditional_value_at_risk_mc_pdf(VaRs, x_cond, alpha=alpha, n_samples=n_samples)
    elif self.can_sample:
      CVaRs = self._conditional_value_at_risk_sampling(VaRs, x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError("Distribution object must either support pdf or sampling in order to compute CVaR")

    assert VaRs.shape == CVaRs.shape == (len(x_cond),)
    return VaRs, CVaRs

