from sklearn.base import BaseEstimator

from cde.utils.integration import mc_integration_student_t, numeric_integation
from cde.utils.center_point_select import *
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import scipy
from cde.utils.optimizers import find_root_newton_method, find_root_by_bounding

""" Default Numerical Integration Standards"""
N_SAMPLES_INT = 10**5
N_SAMPLES_INT_TIGHT_BOUNDS = 10**4
LOWER_BOUND = - 10 ** 3
UPPER_BOUND = 10 ** 3

""" Default Monte-Carlo Integration Standards"""
DOF = 6
LOC_PROPOSAL = 0
SCALE_PROPOSAL = 2

class ConditionalDensity(BaseEstimator):

  """ MEAN """

  def _mean_mc(self, x_cond, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, samples = sample(x)
      means[i, :] = np.mean(samples, axis=0)
    return means

  def _mean_pdf(self, x_cond, n_samples=10 ** 6):
    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      mean_fun = lambda y: y
      if self.ndim_y == 1:
        n_samples_int, lower, upper = self._determine_integration_bounds()
        func_to_integrate = lambda y:  mean_fun(y) * np.squeeze(self._tiled_pdf(y, x_cond[i], n_samples_int))
        integral = numeric_integation(func_to_integrate, n_samples_int, lower, upper)
      else:
        loc_proposal, scale_proposal = self._determine_mc_proposal_dist()
        func_to_integrate = lambda y: mean_fun(y) * self._tiled_pdf(y, x_cond[i], n_samples)
        integral = mc_integration_student_t(func_to_integrate, ndim=self.ndim_y, n_samples=n_samples,
                                            loc_proposal=loc_proposal, scale_proposal=scale_proposal)
      means[i] = integral
    return means

  """ STANDARD DEVIATION """

  def _std_pdf(self, x_cond, n_samples=10**6, mean=None):
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")

    if mean is None:
      mean = self.mean_(x_cond, n_samples=n_samples)

    if self.ndim_y == 1: # compute with numerical integration
      stds = np.zeros((x_cond.shape[0], self.ndim_y))
      for i in range(x_cond.shape[0]):
        mu = np.squeeze(mean[i])
        n_samples_int, lower, upper = self._determine_integration_bounds()
        func_to_integrate = lambda y: (y-mu)**2 * np.squeeze(self._tiled_pdf(y, x_cond[i], n_samples_int))
        stds[i] = np.sqrt(numeric_integation(func_to_integrate, n_samples_int, lower, upper))
    else: # call covariance and return sqrt of diagonal
      covs = self.covariance(x_cond, n_samples=n_samples)
      stds = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))

    return stds

  def _std_mc(self, x_cond, n_samples=10**6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    stds = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, samples = sample(x)
      stds[i, :] = np.std(samples, axis=0)
    return stds

  """ COVARIANCE """

  def _covariance_pdf(self, x_cond, n_samples=10 ** 6, mean=None):
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")
    assert mean is None or mean.shape == (x_cond.shape[0], self.ndim_y)

    loc_proposal, scale_proposal = self._determine_mc_proposal_dist()

    if mean is None:
      mean = self.mean_(x_cond, n_samples=n_samples)

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))

      def cov(y):
        a = (y - mean[i])

        # compute cov matrices c for sampled instances and weight them with the probability p from the pdf
        c = np.empty((a.shape[0], a.shape[1] ** 2))
        for j in range(a.shape[0]):
          c[j, :] = np.reshape(np.outer(a[j], a[j]), (a.shape[1] ** 2,))

        p = np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y ** 2))
        res = c * p
        return res

      integral = mc_integration_student_t(cov, ndim=self.ndim_y, n_samples=n_samples,
                                          loc_proposal=loc_proposal, scale_proposal=scale_proposal)
      covs[i] = integral.reshape((self.ndim_y, self.ndim_y))
    return covs

  def _covariance_mc(self, x_cond, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, y_sample = sample(x)

      c = np.cov(y_sample, rowvar=False)
      covs[i] = c
    return covs

  """ SKEWNESS """

  def _skewness_pdf(self, x_cond, n_samples=10 ** 6, mean=None, std=None):
    assert self.ndim_y == 1, "this function does not support co-skewness - target variable y must be one-dimensional"
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")
    assert hasattr(self, "covariance")

    if mean is None:
      mean = np.reshape(self.mean_(x_cond, n_samples), (x_cond.shape[0],))
    if std is None:
      std = np.reshape(np.sqrt(self.covariance(x_cond, n_samples=n_samples)), (x_cond.shape[0],))

    skewness = np.empty(shape=(x_cond.shape[0],))
    n_samples_int, lower, upper = self._determine_integration_bounds()

    for i in range(x_cond.shape[0]):
      mu = np.squeeze(mean[i])
      sigm = np.squeeze(std[i])
      func_skew = lambda y: ((y - mu) / sigm)**3 * np.squeeze(self._tiled_pdf(y, x_cond[i], n_samples_int))
      skewness[i] = numeric_integation(func_skew, n_samples=n_samples_int)

    return skewness

  def _skewness_mc(self, x_cond, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    skewness = np.empty(shape=(x_cond.shape[0],))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, y_sample = sample(x)

      skewness[i] = scipy.stats.skew(y_sample)
    return skewness

  """ KURTOSIS """

  def _kurtosis_pdf(self, x_cond, n_samples=10 ** 6, mean=None, std=None):
    assert self.ndim_y == 1, "this function does not support co-kurtosis - target variable y must be one-dimensional"
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")
    assert hasattr(self, "covariance")

    if mean is None:
      mean = np.reshape(self.mean_(x_cond, n_samples), (x_cond.shape[0],))
    if std is None:
      std = np.reshape(np.sqrt(self.covariance(x_cond, n_samples=n_samples)), (x_cond.shape[0],))

    n_samples_int, lower, upper = self._determine_integration_bounds()
    kurtosis = np.empty(shape=(x_cond.shape[0],))

    for i in range(x_cond.shape[0]):
      mu = np.squeeze(mean[i])
      sigm = np.squeeze(std[i])
      func_skew = lambda y: ((y - mu)**4 / sigm**4) * np.squeeze(self._tiled_pdf(y, x_cond[i], n_samples_int))
      kurtosis[i] = numeric_integation(func_skew, n_samples=n_samples_int)

    return kurtosis - 3 # excess kurtosis

  def _kurtosis_mc(self, x_cond, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    kurtosis = np.empty(shape=(x_cond.shape[0],))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      _, y_sample = sample(x)

      kurtosis[i] = scipy.stats.kurtosis(y_sample)
    return kurtosis

  """ QUANTILES / VALUE-AT-RISK """

  def _quantile_mc(self, x_cond, alpha=0.01, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    assert x_cond.ndim == 2
    VaRs = np.zeros(x_cond.shape[0])
    x_cond = np.tile(x_cond.reshape((1, x_cond.shape[0], x_cond.shape[1])), (n_samples,1, 1))
    for i in range(x_cond.shape[1]):
      _, samples = sample(x_cond[:, i,:])
      VaRs[i] = np.percentile(samples, alpha * 100.0)
    return VaRs

  def _quantile_cdf(self, x_cond, alpha=0.01, eps=1e-8, init_bound=1e3):
    # finds the alpha quantile of the distribution through root finding by bounding

    cdf_fun = lambda y: self.cdf(x_cond, y) - alpha
    init_bound = init_bound * np.ones(x_cond.shape[0])
    return find_root_by_bounding(cdf_fun, left=-init_bound, right=init_bound, eps=eps)

  """ CONDITONAL VALUE-AT-RISK """

  def _conditional_value_at_risk_mc_pdf(self, VaRs, x_cond, alpha=0.01, n_samples=10 ** 6):
    assert VaRs.shape[0] == x_cond.shape[0], "same number of x_cond must match the number of values_at_risk provided"
    assert self.ndim_y == 1, 'this function only supports only ndim_y = 1'
    assert x_cond.ndim == 2

    n_samples_int, lower, _ = self._determine_integration_bounds()

    CVaRs = np.zeros(x_cond.shape[0])

    for i in range(x_cond.shape[0]):
      upper = float(VaRs[i])
      func_to_integrate = lambda y: y * np.squeeze(self._tiled_pdf(y, x_cond[i], n_samples_int))
      integral = numeric_integation(func_to_integrate, n_samples_int, lower, upper)
      CVaRs[i] = integral / alpha

    return CVaRs

  def _conditional_value_at_risk_sampling(self, VaRs, x_cond, n_samples=10 ** 6):
    if hasattr(self, 'sample'):
      sample = self.sample
    elif hasattr(self, 'simulate_conditional'):
      sample = self.simulate_conditional
    else:
      raise AssertionError("Requires sample or simulate_conditional method")

    CVaRs = np.zeros(x_cond.shape[0])
    x_cond = np.tile(x_cond.reshape((1, x_cond.shape[0], x_cond.shape[1])), (n_samples, 1, 1))
    for i in range(x_cond.shape[1]):
      _, samples = sample(x_cond[:, i, :])
      shortfall_samples = np.ma.masked_where(VaRs[i] < samples, samples)
      CVaRs[i] = np.mean(shortfall_samples)

    return CVaRs

  """ OTHER HELPERS """

  def _handle_input_dimensionality(self, X, Y=None, fitting=False):
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)

    if Y is not None:
      if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

      assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
      assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

    if fitting:  # store n_dim of training data
      self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
    else:
      assert X.shape[1] == self.ndim_x, "X must have shape (?, %i) but provided X has shape %s" % (self.ndim_x, X.shape)
      if Y is not None:
        assert Y.shape[1] == self.ndim_y, "Y must have shape (?, %i) but provided Y has shape %s" % (
        self.ndim_y, Y.shape)

    if Y is None:
      return X
    else:
      return X, Y

  def plot2d(self, x_cond=[0, 1, 2], ylim=(-8, 8), resolution=100, mode='pdf', show=True, prefix='', numpyfig=False):
    """ Generates a 3d surface plot of the fitted conditional distribution if x and y are 1-dimensional each

        Args:
          xlim: 2-tuple specifying the x axis limits
          ylim: 2-tuple specifying the y axis limits
          resolution: integer specifying the resolution of plot
        """
    assert self.ndim_y == 1, "Can only plot two dimensional distributions"
    # prepare mesh

    # turn off interactive mode is show is set to False
    if show == False and mpl.is_interactive():
      plt.ioff()
      mpl.use('Agg')

    fig = plt.figure(dpi=300)
    labels = []

    for i in range(len(x_cond)):
      Y = np.linspace(ylim[0], ylim[1], num=resolution)
      X = np.array([x_cond[i] for _ in range(resolution)])
    # calculate values of distribution

      if mode == "pdf":
        Z = self.pdf(X, Y)
      elif mode == "cdf":
        Z = self.cdf(X, Y)
      elif mode == "joint_pdf":
        Z = self.joint_pdf(X, Y)


      label = "x="+ str(x_cond[i])  if self.ndim_x > 1 else 'x=%.2f' % x_cond[i]
      labels.append(label)

      plt_out = plt.plot(Y, Z, label=label)

    plt.legend([prefix + label for label in labels], loc='upper right')

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

  def plot3d(self, xlim=(-5, 5), ylim=(-8, 8), resolution=100, show=False, numpyfig=False):
    """ Generates a 3d surface plot of the fitted conditional distribution if x and y are 1-dimensional each

    Args:
      xlim: 2-tuple specifying the x axis limits
      ylim: 2-tuple specifying the y axis limits
      resolution: integer specifying the resolution of plot
    """
    assert self.ndim_x + self.ndim_y == 2, "Can only plot two dimensional distributions"

    if show == False and mpl.is_interactive():
      plt.ioff()
      mpl.use('Agg')

    # prepare mesh
    linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
    linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()

    # calculate values of distribution
    Z = self.pdf(X, Y)

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

  def _determine_integration_bounds(self):
    if hasattr(self, 'y_std') and hasattr(self, 'y_mean'):
      lower = self.y_mean - 10 * self.y_std
      upper = self.y_mean + 10 * self.y_std

      return N_SAMPLES_INT_TIGHT_BOUNDS, lower, upper
    else:
      return N_SAMPLES_INT, LOWER_BOUND, UPPER_BOUND

  def _determine_mc_proposal_dist(self):
    if hasattr(self, 'y_std') and hasattr(self, 'y_mean'):
      mu_proposal = self.y_mean
      std_proposal = 1 * self.y_std
      return mu_proposal, std_proposal
    else:
      return np.ones(self.ndim_y) * LOC_PROPOSAL, np.ones(self.ndim_y) * SCALE_PROPOSAL

  def _tiled_pdf(self, Y, x_cond, n_samples):
    x = np.tile(x_cond.reshape((1, x_cond.shape[0])), (n_samples, 1))
    return np.tile(np.expand_dims(self.pdf(x, Y), axis=1), (1, self.ndim_y))