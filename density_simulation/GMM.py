import numpy as np
from sklearn import mixture
from pprint import pprint
import matplotlib.pyplot as plt


np.random.seed(1)


class GMM_Simulation:

  def __init__(self, n_kernels=5, n_dims=2, means_std=10):
    """
    :param n_kernels: number of Gaussian kernels
    :param n_dims: number of dimensions
    :param means_std: std when sampling kernel means
    """

    # sample gmm parameters
    self.weights = np.random.uniform(0, 1, size=[n_kernels])
    self.weights = self.weights / np.sum(self.weights)

    self.means = np.random.normal(loc=np.zeros([n_dims]), scale=means_std, size=[n_kernels, n_dims])
    self.covariances = np.random.uniform(low=0.1, high=10, size=[n_kernels, n_dims])

    self.gmm = mixture.GaussianMixture(n_components=n_kernels, covariance_type='diag')


    self.gmm = self.gmm.fit(np.random.normal(size=[2000, n_dims]))
    self.gmm.weights_ = self.weights
    self.gmm.means_ = self.means
    self.gmm.covariances_ = self.covariances


  def sample(self, n_samples=1):
    return self.gmm.sample(n_samples=n_samples)


def main():
  gmm = GMM_Simulation(n_kernels=4)
  sim_data = gmm.sample(n_samples=1000)[0]
  print(gmm.means)
  plt.scatter(x = sim_data[:,0], y=sim_data[:,1])
  plt.show()




if __name__ == "__main__":
  main()