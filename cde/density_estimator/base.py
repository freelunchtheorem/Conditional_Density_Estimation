from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import warnings
from scipy.stats import multivariate_normal
#matplotlib.use("PS") #handles X11 server detection (required to run on console)
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm


class BaseDensityEstimator(BaseEstimator):
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
        resulution: integer specifying the resolution of evaluation grid

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


  def fit_by_cv(self, X, Y, n_folds=5, param_grid=None):
    """ Fits the conditional density model with hyperparameter search and cross-validation.

    - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
      the conditional log-likelihood is used for evaluation.
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


  # def plot(self, xlim=(0, 3.5), ylim=(0, 8), resolution=50):
  #   """ Plots the fitted conditional distribution if x and y are 1-dimensional each
  #
  #   Args:
  #     xlim: 2-tuple specifying the x axis limits
  #     ylim: 2-tuple specifying the y axis limits
  #     resolution: integer specifying the resolution of plot
  #   """
  #   assert self.fitted, "model must be fitted to plot"
  #   assert self.ndim_x + self.ndim_y == 2, "Can only plot two dimensional distributions"
  #
  #   # prepare mesh
  #   linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
  #   linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
  #   X, Y = np.meshgrid(linspace_x, linspace_y)
  #   X, Y = X.flatten(), Y.flatten()
  #
  #   # calculate values of distribution
  #   Z = self.pdf(X, Y)
  #
  #   X, Y, Z = X.reshape([resolution, resolution]), Y.reshape([resolution, resolution]), Z.reshape(
  #     [resolution, resolution])
  #   fig = plt.figure()
  #   ax = fig.gca(projection='3d')
  #   surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=resolution, ccount=resolution,
  #                          linewidth=100, antialiased=True)
  #   plt.xlabel("x")
  #   plt.ylabel("y")
  #   plt.show()
  #

  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    """ Converts data arrays into 2d numpy arrays

    Args:
      X: numpy array of shape (n_samples,) or (n_samples, ndim_x)
      Y: (optional) numpy array of shape (n_samples,) or (n_samples, ndim_y)
      fitting: boolean that indicates whether the method is called within the model fitting process. /
                      if true, the shapes of the inputs X and Y are stored in the model instance

    Returns:
      If Y was provided - returns (X, Y) where both X and Y are 2d numpy arrays, else only returns X as 2d numpy array
    """

    if np.size(X) == 1 and Y is None:
      return X

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)

    if Y is not None:
      if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

      assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
      assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    if fitting: # store n_dim of training data
      self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    else:
      assert X.shape[1] == self.ndim_x, "X must have shape (?, %i) but provided X has shape %s" % (self.ndim_x, X.shape)
      if Y is not None:
        assert Y.shape[1] == self.ndim_y, "Y must have shape (?, %i) but provided Y has shape %s" % (self.ndim_y, Y.shape)

    if Y is None:
      return X
    else:
      return X, Y


  def __str__(self):
    raise NotImplementedError


  def get_params(self, deep=True):
    """ Get parameters for this estimator.

    Args:
      deep: boolean, optional If True, will return the parameters for this estimator and \
             contained subobjects that are estimators.

    Returns:
      params - mapping of string to any Parameter names mapped to their values.

    """
    param_dict = super(BaseDensityEstimator, self).get_params(deep=deep)
    #param_dict['estimator'] = self.__class__.__name__
    return param_dict

  def value_at_risk(self, x_cond, alpha=0.01):
    """ Computes the Value-at-Risk of the fitted distribution. Only if ndim_y = 1

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, )
      alpha: quantile percentage of the distribution

    Returns:
       x values to condition on - numpy array of shape (n_values)
    """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 1

    if self.has_cdf:
      return self._value_at_risk_cdf(x_cond, alpha=alpha)
    elif self.can_sample:
      return self._value_at_risk_mc(x_cond, alpha=alpha)
    else:
      raise NotImplementedError()

  def _value_at_risk_mc(self, x_cond, alpha=0.01, n_samples=10**7):
    VaRs = np.zeros(x_cond.shape)
    x_cond = np.tile(x_cond.reshape((1, x_cond.shape[0])), (n_samples,1))
    for i in range(x_cond.shape[1]):
      _, samples = self.sample(x_cond[:,i])
      VaRs[i] = np.percentile(samples, alpha * 100.0)
    return VaRs

  def _value_at_risk_cdf(self, x_cond, alpha=0.01, eps=10**-8):
    approx_error = 10**8
    x = x_cond.reshape((1, x_cond.shape[0]))
    left, right = -10**8, 10**8

    VaRs = np.zeros(x_cond.shape)

    # numerical approximation, i.e. Newton method
    for j in range(x.shape[1]):
      while approx_error > eps:
        middle = (left+right) / 2
        y = np.array([middle])
        p = self.cdf(x[:,j],y)

        if p > alpha:
          right = middle
        else:
          left = middle
        approx_error = np.abs(p - alpha)
      VaRs[j] = middle
    return VaRs

class BaseMixtureEstimator(BaseDensityEstimator):

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