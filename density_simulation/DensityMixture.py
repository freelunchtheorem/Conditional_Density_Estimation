import numpy as np
from sklearn import mixture
from pprint import pprint
import matplotlib.pyplot as plt
import pomegranate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats



np.random.seed(1)


class GMM_Simulation:
  def __init__(self, n_kernels=5, n_dims=2, means_std=10):
    """
    :param n_kernels: number of Gaussian kernels
    :param n_dims: number of dimensions
    :param means_std: std when sampling kernel means
    """

    # sample gmm parameters
    self.weights = sample_weights(n_kernels)

    self.means = np.random.normal(loc=np.zeros([n_dims]), scale=means_std, size=[n_kernels, n_dims])
    self.covariances = np.diag(np.random.uniform(low=0.1, high=10, size=[n_kernels, n_dims]))

    self.gmm = mixture.GaussianMixture(n_components=n_kernels, covariance_type='diag')

    self.gmm = self.gmm.fit(np.random.normal(size=[2000, n_dims]))
    self.gmm.weights_ = self.weights
    self.gmm.means_ = self.means
    self.gmm.covariances_ = self.covariances

  def sample(self, n_samples=1):
    """
    :param n_samples: number of samples
    :return: sampled data - numpy array with shape (n_samples, n_dims)
    """
    return self.gmm.sample(n_samples=n_samples)


# class DensityMixture:
#
#   def __init__(self):
#     d1 = pomegranate.MultivariateGaussianDistribution([1, 4], [[1, 0.4], [0.4, 1]])
#     d2 = pomegranate.DirichletDistribution([10, 8], [[4, 0.2], [0.2, 0.1]])
#     d3 = pomegranate.MultivariateGaussianDistribution([-3, 2], [[1, 0], [0, 1]])
#     self.model = pomegranate.GeneralMixtureModel([d1, d2, d3])
#
#   def sample(self, n_samples=1):
#     """
#     :param n_samples: number of samples
#     :return: sampled data - numpy array with shape (n_samples, n_dims)
#     """
#     return np.stack(self.model.sample(n=n_samples), axis=0)

class GeneralDensity:

  def plot_density(self, xlim=(-5,5), ylim=(-5,5)):
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

class DensityMixture(GeneralDensity):

  def __init__(self):
    self.weights = sample_weights(3)


  def pdf(self, pos):
    mu1 = np.array([0., 1.])
    sigma1 = np.array([[4., -0.5], [-0.5, 1.5]])
    F1 = stats.multivariate_normal(mu1, sigma1)

    mu2 = np.array([-3., 4])
    sigma2 = np.array([[2.2, 0.3], [0.3, 0.9]])
    F2 = stats.multivariate_normal(mu2, sigma2)

    return self.weights[0] * F1.pdf(pos) + self.weights[1] * F2.pdf(pos)


def sample_weights(n_weights):
  """
  samples density weights -> sum up to one
  :param n_weights: number of weights
  :return: ndarray of weights with shape (n_weights,)
  """
  weights = np.random.uniform(0, 1, size=[n_weights])
  return weights / np.sum(weights)


def main():
  gd = DensityMixture()
  gd.plot_density()



if __name__ == "__main__":
  main()