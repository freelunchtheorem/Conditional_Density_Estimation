from sklearn.model_selection import GridSearchCV
import warnings
from scipy.stats import multivariate_normal, norm
#matplotlib.use("PS") #handles X11 server detection (required to run on console)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.mixture import GaussianMixture

from cde import ConditionalDensity

from cde.helpers import *

class BaseDensityEstimator(ConditionalDensity):
  """ Interface for conditional density estimation models """

  def fit(self, X, Y):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        n_folds: number of cross-validation folds (positive integer)

    """
    raise NotImplementedError

  def pdf(self, X, Y):
    """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )

     """
    raise NotImplementedError

  def predict_density(self, X, Y=None, resolution=50):
    """ Computes conditional density p(y|x) over a predefined grid of y target values

      Args:
        X: values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
        Y: (optional) y values to be evaluated from p(y|x) -  if not set, Y will be a grid with with specified resolution
        resulution: integer specifying the resolution of evaluation_runs grid

      Returns: tuple (P, Y)
         - P - density p(y|x) - shape (n_instances, resolution**n_dim_y)
         - Y - grid with with specified resolution - shape (resolution**n_dim_y, n_dim_y) or a copy of Y \
           in case it was provided as argument
    """
    raise NotImplementedError

  def _param_grid(self):
    raise NotImplementedError

  def score(self, X, Y):
    """Computes the mean conditional log-likelihood of the provided data (X, Y)

    Args:
      X: numpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
      Y: numpy array of y targets - shape: (n_query_samples, n_dim_y)

    Returns:
      negative log likelihood of data
    """
    if hasattr(self, 'log_pdf'):
      return(np.mean(self.log_pdf(X, Y)))
    else:
      X, Y = self._handle_input_dimensionality(X, Y, fitting=False)

      assert self.fitted, "model must be fitted to compute likelihood score"
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # don't print division by zero warning
        conditional_log_likelihoods = np.log(self.pdf(X, Y))
      return np.mean(conditional_log_likelihoods)

  def mean_(self, x_cond, n_samples=10**7):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    assert x_cond.ndim == 2

    if self.can_sample:
      return self._mean_mc(x_cond, n_samples=n_samples)
    else:
      return self._mean_pdf(x_cond)

  def covariance(self, x_cond, n_samples=10**7):
    """ Covariance of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    return self._covariance_pdf(x_cond, n_samples=n_samples)

  def value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
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

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
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

  def fit_by_cv(self, X, Y, n_folds=5, param_grid=None):
    """ Fits the conditional density model with hyperparameter search and cross-validation.

    - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
      the conditional log-likelihood is used for evaluation_runs.
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

    cv_model = GridSearchCV(self, param_grid, fit_params=None, n_jobs=-1, refit=True, cv=n_folds,
                 verbose=1)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # don't print division by zero warning
      cv_model.fit(X,Y)
    best_params = cv_model.best_params_
    print("Cross-Validation terminated")
    print("Best likelihood score: %.4f"%cv_model.best_score_)
    print("Best params:", best_params)
    self.set_params(**cv_model.best_params_)
    self.fit(X,Y)

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

  def plot3d(self, xlim=(0, 3.5), ylim=(0, 8), resolution=50):
    """ Generates a 3d surface plot of the fitted conditional distribution if x and y are 1-dimensional each

    Args:
      xlim: 2-tuple specifying the x axis limits
      ylim: 2-tuple specifying the y axis limits
      resolution: integer specifying the resolution of plot
    """
    assert self.fitted, "model must be fitted to plot"
    assert self.ndim_x + self.ndim_y == 2, "Can only plot two dimensional distributions"

    # prepare mesh
    linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
    linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()

    # calculate values of distribution
    Z = self.pdf(X, Y)

    X, Y, Z = X.reshape([resolution, resolution]), Y.reshape([resolution, resolution]), Z.reshape(
      [resolution, resolution])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=resolution, ccount=resolution,
                           linewidth=100, antialiased=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 7):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo evaluation

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

  def __str__(self):
    raise NotImplementedError


class BaseMixtureEstimator(BaseDensityEstimator):

  def mean_(self, x_cond, n_samples=10**7):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert hasattr(self, '_get_mixture_components')
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    means = np.zeros((x_cond.shape[0], self.ndim_y))
    weights, locs, _ = self._get_mixture_components(x_cond)
    assert weights.ndim == 2 and locs.ndim == 3
    for i in range(x_cond.shape[0]):
      # mean of density mixture is weights * means of density components
      means[i, :] = weights[i].dot(locs[i])
    return means

  def covariance(self, x_cond, n_samples=None):
    """ Covariance of the fitted distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert self.fitted, "model must be fitted"
    x_cond = self._handle_input_dimensionality(x_cond)
    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))

    # compute global mean_of mixture model
    glob_mean = self.mean_(x_cond)

    weights, locs, scales = self._get_mixture_components(x_cond)

    for i in range(x_cond.shape[0]):
      c1 = np.diag(weights[i].dot(scales[i]))

      c2 = np.zeros(c1.shape)
      for j in range(weights.shape[0]):
        a = (locs[i][j] - glob_mean[i])
        d = weights[i][j] * np.outer(a,a)
        c2 += d
      covs[i] = c1 + c2

    return covs

  def sample(self, X):
    """ sample from the conditional mixture distributions - requires the model to be fitted

      Args:
        X: values to be conditioned on when sampling - numpy array of shape (n_instances, n_dim_x)

      Returns: tuple (X, Y)
        - X - the values to conditioned on that were provided as argument - numpy array of shape (n_samples, ndim_x)
        - Y - conditional samples from the model p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    assert self.can_sample

    X = self._handle_input_dimensionality(X)

    if np.all(np.all(X == X[0, :], axis=1)):
      return self._sample_rows_same(X)
    else:
      return self._sample_rows_individually(X)

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of a GMM. Only if ndim_y = 1

        Based on formulas from section 2.3.2 in "Expected shortfall for distributions in finance",
        Simon A. Broda, Marc S. Paolella, 2011

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
    return self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)

  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 7):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo evaluation

        Returns:
          - VaR values for each x to condition on - numpy array of shape (n_values)
          - CVaR values for each x to condition on - numpy array of shape (n_values)
        """
    assert self.fitted, "model must be fitted"
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
    CVaRs = self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)

    assert VaRs.shape == CVaRs.shape == (len(x_cond),)
    return VaRs, CVaRs

  def _conditional_value_at_risk_mixture(self, VaRs, x_cond, alpha=0.01,):
    """
    Based on formulas from section 2.3.2 in "Expected shortfall for distributions in finance",
    Simon A. Broda, Marc S. Paolella, 2011
    """

    weights, locs, scales = self._get_mixture_components(x_cond)

    locs = locs.reshape(locs.shape[:2])
    scales = scales.reshape(scales.shape[:2])

    CVaRs = np.zeros(x_cond.shape[0])

    c = (VaRs[:, None] - locs) / scales
    for i in range(x_cond.shape[0]):
      cdf = norm.cdf(c[i])
      pdf = norm.pdf(c[i])

      # mask very small values to avoid numerical instabilities
      cdf = np.ma.masked_where(cdf < 10 ** -64, cdf)
      pdf = np.ma.masked_where(pdf < 10 ** -64, pdf)

      CVaRs[i] = np.sum((weights[i] * cdf / alpha) * (locs[i] - scales[i] * (pdf / cdf)))
    return CVaRs

  def _sample_rows_same(self, X):
    """ uses efficient sklearn implementation to sample from gaussian mixture -> only works if all rows of X are the same"""
    weights, locs, scales = self._get_mixture_components(np.expand_dims(X[0], axis=0))

    # make sure that sum of weights < 1
    weights = weights.astype(np.float64)
    weights = weights / np.sum(weights)

    gmm = GaussianMixture(n_components=self.n_centers, covariance_type='diag', max_iter=5, tol=1e-1)
    gmm.fit(np.random.normal(size=(100,self.ndim_y))) # just pretending a fit
    # overriding the GMM parameters with own params
    gmm.converged_ = True
    gmm.weights_ = weights[0]
    gmm.means_ = locs[0]
    gmm.covariances_ = scales[0]
    y_sample, _ = gmm.sample(X.shape[0])
    assert y_sample.shape == (X.shape[0], self.ndim_y)
    return X, y_sample

  def _sample_rows_individually(self, X):
    weights, locs, scales = self._get_mixture_components(X)

    Y = np.zeros(shape=(X.shape[0], self.ndim_y))
    for i in range(X.shape[0]):
      idx = np.random.choice(range(self.n_centers), p=weights[i, :])
      Y[i, :] = np.random.normal(loc=locs[i, idx, :], scale=scales[i, idx, :])
    assert X.shape[0] == Y.shape[0]
    return X, Y

  def cdf(self, X, Y):
    """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
         conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )

    """
    assert self.fitted, "model must be fitted to compute likelihood score"
    assert hasattr(self, '_get_mixture_components'), "cdf computation requires _get_mixture_components method"

    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)

    weights, locs, scales = self._get_mixture_components(X)

    P = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      for j in range(self.n_centers):
        P[i] += weights[i, j] * multivariate_normal.cdf(Y[i], mean=locs[i,j,:], cov=np.diag(scales[i,j,:]))
    return P