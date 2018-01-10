import numpy as np
from sklearn import mixture
from pprint import pprint
import matplotlib.pyplot as plt
import pomegranate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from density_simulation import ConditionalDensity
from density_estimator import helpers



np.random.seed(1)


class GMM(ConditionalDensity):
  """
  A gaussian mixture model for drawing conditional samples from its mixture distribution. Implements the
  ConditionDensity class.
  """
  def __init__(self, n_kernels=5, ndim_x=1, ndim_y=1, means_std=10):
    """
    :param n_kernels: number of mixture components
    :param ndim_x: dimensionality of X / number of random variables in X
    :param ndim_y: dimensionality of Y / number of random variables in Y
    :param n_dims: sum of X and Y dimensions
    :param means_std: std. dev. when sampling the kernel means
    """

    """  set parameters, calculate weights, means and covariances """
    self.n_kernels = 5
    self.n_dims = ndim_x + ndim_y
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.weights = self.sample_weights(n_kernels) #shape(n_kernels,), sums to one
    self.means = np.random.normal(loc=np.zeros([self.n_dims]), scale=means_std, size=[n_kernels, self.n_dims]) #shape(n_kernels, n_dims)
    self.covariances = np.random.uniform(low=0.1, high=2, size=(n_kernels, self.n_dims, self.n_dims)) #shape(n_kernels, ndim_x, ndim_y)


    """ some eigenvalues of the sampled covariance matrices can be exactly zero -> map to positive 
    semi-definite subspace  """
    self.covariances = helpers.project_to_pos_semi_def(self.covariances)


    """ after mapping, define the remaining variables and collect frozen multivariate variables
      (x,y), x and y for later conditional draws """
    self.means_x = self.means[:, :ndim_x]
    self.covariances_x = self.covariances[:, :ndim_x, :ndim_x]
    self.means_y = self.means[:, ndim_x:]
    self.covariances_y = self.covariances[:, ndim_x:, ndim_x:]

    self.gaussians, self.gaussians_x, self.gaussians_y = [], [], []
    for i in range(n_kernels):
      self.gaussians.append(stats.multivariate_normal(mean=self.means[i,], cov=self.covariances[i]))
      self.gaussians_x.append(stats.multivariate_normal(mean=self.means_x[i,], cov=self.covariances_x[i]))
      self.gaussians_y.append(stats.multivariate_normal(mean=self.means_y[i,], cov=self.covariances_y[i]))



  def cdf(self, X, Y):
    # todo
    raise NotImplementedError


  def pdf(self, X, Y):
    """
    Determines the conditional probability density function P(Y|X).
    See "Conditional Gaussian Mixture Models for Environmental Risk Mapping" [Gilardi, Bengio] for the math.
    :param X: the position/conditional random variable for the distribution P(Y|X), array_like, shape:(n_samples,
    ndim_x)
    :param Y: the on X conditioned random variable Y, array_like, shape:(n_samples, ndim_y)
    :return: the cond. distribution of Y given X, for the given realizations of X with shape:(n_samples,)
    """
    assert X.shape[1] == self.ndim_x
    assert Y.shape[1] == self.ndim_y
    assert X.shape[0] == Y.shape[0]

    P_y = np.stack([self.weights[i] * self.gaussians_y[i].pdf(X) for i in range(self.n_kernels)], axis=1) #shape(X.shape[0], n_kernels)
    W_x = self._W_x(X)

    cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)
    assert cond_prob.shape[0] == X.shape[0]
    return cond_prob



  def _W_x(self, X):
    """
    Helper function to normalize the joint density P(Y,X) by the marginal density P(X)
    :param X: conditional random variable, array_like, shape:(n_samples, ndim_x)
    :return: the normalized weighted marginal gaussian distributions P(X) for each n_kernel, shape:(n_samples,n_kernels)
    """
    w_p = np.stack([self.weights[i] * self.gaussians_x[i].pdf(X) for i in range(self.n_kernels)], axis=1)
    normalizing_term = np.sum(w_p, axis=1)
    result = w_p / normalizing_term[:,None]
    return result



  def simulate_conditional(self, X):
    # todo
    raise NotImplementedError



  def simulate(self, n_samples=1000):
    """
    this draws (n_samples) instances from the (n_kernels)-multivariate normal distributions
    :param n_samples:
    :return:
    """
    assert n_samples > 0
    discrete_dist = stats.rv_discrete(values=(range(self.n_kernels), self.weights))
    indices = discrete_dist.rvs(size=n_samples)

    draw_sample = lambda i: self.gaussians[i].rvs()

    samples_joint = np.stack(list(map(draw_sample, indices)), axis=0)

    x_samples = samples_joint[:, :self.ndim_x]
    y_samples = samples_joint[:, self.ndim_x:]
    assert x_samples.shape == (n_samples, self.ndim_x)
    assert y_samples.shape == (n_samples, self.ndim_y)
    return x_samples, y_samples


  def sample_weights(self, n_weights):
    """
    samples density weights -> sum up to one
    :param n_weights: number of weights
    :return: ndarray of weights with shape (n_weights,)
    """
    weights = np.random.uniform(0, 1, size=[n_weights])
    return weights / np.sum(weights)


class DensityMixtureUnconditional:
  """
  unconditional
  """
  def __init__(self):
    self.weights = self.sample_weights(3)


  def pdf(self, Y):
    mu1 = np.array([0., 1.])
    sigma1 = np.array([[4., -0.5], [-0.5, 1.5]])
    F1 = stats.multivariate_normal(mu1, sigma1)

    mu2 = np.array([-3., 4])
    sigma2 = np.array([[2.2, 0.3], [0.3, 0.9]])
    F2 = stats.multivariate_normal(mu2, sigma2)

    return self.weights[0] * F1.pdf(Y) + self.weights[1] * F2.pdf(Y)



  def plot(self, xlim=(-5,5), ylim=(-5,5)):
    """
    Plots the density function
    :param xlim: 2-tuple with the x-axis limits
    :param ylim: 2-tuple with the y-axis limits
    """
    assert type(xlim) is tuple and type(ylim) is tuple, "xlim / ylim must be a tuple"

    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X, Y = np.linspace(xlim[0], xlim[1], N), np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = self.pdf(pos)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    plt.show()


  def sample_weights(self, n_weights):
    """
    samples density weights -> sum up to one
    :param n_weights: number of weights
    :return: ndarray of weights with shape (n_weights,)
    """
    weights = np.random.uniform(0, 1, size=[n_weights])
    return weights / np.sum(weights)



class EconDensity(ConditionalDensity):
  """
  The economy dataset generated by the function y=x2 + N(0,self.std). Inherits ConditionalDensity class.
  """
  # todo: generalize to ndim_x, ndim_y-usage

  def __init__(self, std):
    self.std = std

  def pdf(self, X, Y):
    mean = X**2
    return stats.norm.pdf(Y, loc=mean, scale=self.std)

  def cdf(self, X, Y):
    mean = X**2
    return stats.norm.cdf(Y, loc=mean, scale=self.std)

  def simulate_conditional(self, X):
    assert X.ndim == 1
    n_samples = X.shape[0]
    Y = X ** 2 + np.random.normal(loc=0, scale=self.std, size=[n_samples])
    return X, Y

  def simulate(self, n_samples=1000):
    assert n_samples > 0
    X = np.abs(np.random.standard_normal(size=[n_samples]))
    Y = X ** 2 + np.random.normal(loc=0, scale=self.std, size=[n_samples])
    return X, Y

  def plot(self, xlim, ylim):
    # todo
    raise NotImplementedError



def main():
  ed = EconDensity(std=2)
  samples = ed.simulate(1000)
  pdf = ed.pdf(samples[0], samples[1])
  samples_cond = ed.simulate_conditional(samples[0])


  #gd = DensityMixtureConditional()
  #gd.plot()

  gmm = GMM(ndim_x=1, ndim_y=1)
  print(gmm.gaussians)

  """ draw from unconditional distribution """
  data_uncond = gmm.simulate(n_samples=1000)

  """ draw from conditional distribution """
  X = data_uncond[0]
  Y = data_uncond[1]
  probs = gmm.pdf(X, Y)
  print(probs.shape)

if __name__ == "__main__":
  main()