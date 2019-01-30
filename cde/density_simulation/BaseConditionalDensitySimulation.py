import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from mpl_toolkits.mplot3d import Axes3D
from cde import ConditionalDensity



class BaseConditionalDensitySimulation(ConditionalDensity):

  def pdf(self, X, Y):
    """ Conditional probability density function p(y|x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the pdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
      p(X|Y) conditional density values for the provided X and Y - numpy array of shape (n_points, )
    """

    raise NotImplementedError

  def log_pdf(self, X, Y):
    """ Conditional log-probability log p(y|x). Requires the model to be fitted.

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

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
     P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
    """

    raise NotImplementedError

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    raise NotImplementedError

  def simulate(self, n_samples):
    """ Draws random samples from the unconditional distribution p(x,y)

    Args:
      n_samples: (int) number of samples to be drawn from the conditional distribution

    Returns:
      (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    raise NotImplementedError

  def plot(self, xlim=(-5, 5), ylim=(-5, 5), resolution=100, mode="pdf", show=False, numpyfig=False):
    """ Plots the distribution specified in mode if x and y are 1-dimensional each

    Args:
      xlim: 2-tuple specifying the x axis limits
      ylim: 2-tuple specifying the y axis limits
      resolution: integer specifying the resolution of plot
      mode: spefify which dist to plot ["pdf", "cdf", "joint_pdf"]

    """
    modes = ["pdf", "cdf", "joint_pdf"]
    assert mode in modes, "mode must be on of the following: " + modes
    assert self.ndim == 2, "Can only plot two dimensional distributions"

    if show == False and mpl.is_interactive():
      plt.ioff()


    # prepare mesh
    linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
    linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()

    # calculate values of distribution
    if mode == "pdf":
      Z = self.pdf(X, Y)
    elif mode == "cdf":
      Z = self.cdf(X, Y)
    elif mode == "joint_pdf":
      Z = self.joint_pdf(X, Y)

    X, Y, Z = X.reshape([resolution, resolution]), Y.reshape([resolution, resolution]), Z.reshape(
      [resolution, resolution])
    fig = plt.figure(dpi=300)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=resolution, ccount=resolution,
                           linewidth=100, antialiased=True)
    plt.xlabel("x")
    plt.ylabel("y")
    if show:
      plt.show()

    if numpyfig:
      fig.tight_layout(pad=0)
      fig.canvas.draw()
      numpy_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
      numpy_img = numpy_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      return numpy_img

    return fig

  def mean_(self, x_cond, n_samples=10**6):
    """ Mean of the fitted distribution conditioned on x_cond
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert x_cond.ndim == 2

    if self.can_sample:
      return self._mean_mc(x_cond, n_samples=n_samples)
    else:
      return self._mean_pdf(x_cond)

  def std_(self, x_cond, n_samples=10 ** 6):
    """ Standard deviation of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Standard deviations  sqrt(Var[y|x]) corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    return self._std_pdf(x_cond, n_samples=n_samples)

  def covariance(self, x_cond, n_samples=10**6):
    """ Covariance of the fitted distribution conditioned on x_cond

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      n_samples: number of samples for monte carlo model_fitting

    Returns:
      Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    if self.has_pdf:
      return self._covariance_pdf(x_cond)
    elif self.can_sample:
      return self._covariance_mc(x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def skewness(self, x_cond, n_samples=10 ** 6):
    """ Skewness of the fitted distribution conditioned on x_cond

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

       Returns:
         Skewness Skew[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
       """
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    if self.has_pdf:
      return self._skewness_pdf(x_cond, n_samples=n_samples)
    elif self.can_sample:
      return self._skewness_pdf(x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def kurtosis(self, x_cond, n_samples=10 ** 6):
    """ Kurtosis of the fitted distribution conditioned on x_cond

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

       Returns:
         Kurtosis Kurt[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
       """
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2
    if self.has_pdf:
      return self._kurtosis_pdf(x_cond, n_samples=n_samples)
    elif self.can_sample:
      return self._kurtosis_mc(x_cond, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def value_at_risk(self, x_cond, alpha=0.01, n_samples=10**6):
    """ Computes the Value-at-Risk (VaR) of the fitted distribution. Only if ndim_y = 1

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      alpha: quantile percentage of the distribution
      n_samples: number of samples for monte carlo model_fitting

    Returns:
       VaR values for each x to condition on - numpy array of shape (n_values)
    """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    if self.has_cdf:
      return self._quantile_cdf(x_cond, alpha=alpha)
    elif self.can_sample:
      return self._quantile_mc(x_cond, alpha=alpha, n_samples=n_samples)
    else:
      raise NotImplementedError()

  def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**6):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of the fitted distribution. Only if ndim_y = 1

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
         alpha: quantile percentage of the distribution
         n_samples: number of samples for monte carlo model_fitting

       Returns:
         CVaR values for each x to condition on - numpy array of shape (n_values)
       """
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

  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10**6):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo model_fitting

        Returns:
          - VaR values for each x to condition on - numpy array of shape (n_values)
          - CVaR values for each x to condition on - numpy array of shape (n_values)
        """
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

  def get_configuration(self, deep=True):
    param_dict = super(BaseConditionalDensitySimulation, self).get_params(deep=deep)
    param_dict['simulator'] = self.__class__.__name__
    return param_dict

  def _handle_input_dimensionality(self, X, Y=None):
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)

    if Y is not None:
      if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

      assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
      assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    if Y is None:
      return X
    else:
      return X, Y

  def _compute_data_statistics(self):
    _, Y = self.simulate(n_samples=10**4)
    return np.mean(Y, axis=0), np.std(Y, axis=0)