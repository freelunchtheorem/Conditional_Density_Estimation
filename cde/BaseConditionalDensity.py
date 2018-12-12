from sklearn.base import BaseEstimator
import numpy as np
from .helpers import *
import scipy.stats as stats
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from cde.utils.optimizers import find_root_newton_method, find_root_by_bounding

class ConditionalDensity(BaseEstimator):

  """ MEAN """

  def _mean_mc(self, x_cond, n_samples=10 ** 7):
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

  def _mean_pdf(self, x_cond, n_samples=10 ** 7):
    means = np.zeros((x_cond.shape[0], self.ndim_y))
    for i in range(x_cond.shape[0]):
      x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
      func = lambda y: y * np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y))
      integral = mc_integration_cauchy(func, ndim=self.ndim_y, n_samples=n_samples)
      means[i] = integral
    return means

  """ COVARIANCE """

  def _covariance_pdf(self, x_cond, n_samples=10 ** 6):
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
    mean = self.mean_(x_cond)
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

      integral = mc_integration_cauchy(cov, ndim=self.ndim_y, n_samples=n_samples)
      covs[i] = integral.reshape((self.ndim_y, self.ndim_y))
    return covs

  def _covariance_mc(self, x_cond, n_samples=10 ** 7):
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

  def _skewness_pdf(self, x_cond, n_samples=10 ** 6):
    assert self.ndim_y == 1, "this function does not support co-skewness - target variable y must be one-dimensional"
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")
    assert hasattr(self, "covariance")

    mean = np.reshape(self.mean_(x_cond, n_samples), (x_cond.shape[0],))
    std = np.reshape(np.sqrt(self.covariance(x_cond, n_samples=n_samples)), (x_cond.shape[0],))

    skewness = np.empty(shape=(x_cond.shape[0],))

    for i in range(x_cond.shape[0]):
      x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))

      def skew(y):
        s = ((y - mean[i]) / std[i])**3
        p = np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y ** 2))
        res = s * p
        return res

      integral = mc_integration_cauchy(skew, ndim=self.ndim_y, n_samples=n_samples)
      skewness[i] = integral.reshape((self.ndim_y, self.ndim_y))

    return skewness

  def _skewness_mc(self, x_cond, n_samples=10 ** 7):
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

  def _kurtosis_pdf(self, x_cond, n_samples=10 ** 6):
    assert self.ndim_y == 1, "this function does not support co-kurtosis - target variable y must be one-dimensional"
    assert hasattr(self, "mean_")
    assert hasattr(self, "pdf")
    assert hasattr(self, "covariance")

    mean = np.reshape(self.mean_(x_cond, n_samples), (x_cond.shape[0],))
    var = np.reshape(self.covariance(x_cond, n_samples=n_samples), (x_cond.shape[0],))

    kurtosis = np.empty(shape=(x_cond.shape[0],))

    for i in range(x_cond.shape[0]):
      x = x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))

      def kurt(y):
        k = (y - mean[i])**4 / var[i]**2
        p = np.tile(np.expand_dims(self.pdf(x, y), axis=1), (1, self.ndim_y ** 2))
        res = k * p
        return res

      integral = mc_integration_cauchy(kurt, ndim=self.ndim_y, n_samples=n_samples)
      kurtosis[i] = integral.reshape((self.ndim_y, self.ndim_y))

    return kurtosis - 3 # excess kurtosis

  def _kurtosis_mc(self, x_cond, n_samples=10 ** 7):
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

  def _quantile_mc(self, x_cond, alpha=0.01, n_samples=10 ** 7):
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

  def _quantile_cdf_old(self, x_cond, alpha=0.01, eps=10 ** -8):
    # Newton Method for finding the alpha quantile of a conditional distribution -> slower than bounding method
    cdf_fun = lambda y: self.cdf(x_cond, y) - alpha
    pdf_fun = lambda y: self.pdf(x_cond, y)

    return find_root_newton_method(fun=cdf_fun, grad=pdf_fun, x0=np.zeros(x_cond.shape[0]), eps=eps, max_iter=max_iter)

  """ CONDITONAL VALUE-AT-RISK """

  def _conditional_value_at_risk_mc_pdf(self, VaRs, x_cond, alpha=0.01, n_samples=10 ** 7):
    assert VaRs.shape[0] == x_cond.shape[0], "same number of x_cond must match the number of values_at_risk provided"
    assert x_cond.ndim == 2

    CVaRs = np.zeros(x_cond.shape[0])

    # preparations for importance sampling from exponential distribtution
    scale = 0.4 # 1 \ lambda
    sampling_dist = stats.expon(scale=scale)
    exp_samples = sampling_dist.rvs(size=n_samples).flatten()
    exp_f = sampling_dist.pdf(exp_samples)  #1 / scale * np.exp(-exp_samples/scale)

    # check shapes
    assert exp_samples.shape[0] == exp_f.shape[0] == n_samples

    for i in range(x_cond.shape[0]):
      # flip the normal exponential distribution by negating it & placing it's mode at the VaR value
      y_samples = VaRs[i] - exp_samples

      x_cond_tiled = np.tile(np.expand_dims(x_cond[i,:], axis=0), (n_samples, 1))
      assert x_cond_tiled.shape == (n_samples, self.ndim_x)

      p = self.pdf(x_cond_tiled, y_samples).flatten()
      q = exp_f.flatten()
      importance_weights = p / q
      cvar = np.mean(y_samples * importance_weights, axis=0) / alpha
      CVaRs[i] = cvar

    return CVaRs

  def _conditional_value_at_risk_sampling(self, VaRs, x_cond, n_samples=10 ** 7):
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

