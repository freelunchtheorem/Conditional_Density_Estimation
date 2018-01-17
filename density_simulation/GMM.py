from density_simulation import ConditionalDensity
import numpy as np
import density_estimator.helpers as helpers
import scipy.stats as stats
from density_simulation import ConditionalDensity
from density_estimator import helpers

class GaussianMixture(ConditionalDensity):
  """
  A gaussian mixture model for drawing conditional samples from its mixture distribution. Implements the
  ConditionDensity class.
  """
  def __init__(self, n_kernels=5, ndim_x=1, ndim_y=1, means_std=1.5):
    """
    :param n_kernels: number of mixture components
    :param ndim_x: dimensionality of X / number of random variables in X
    :param ndim_y: dimensionality of Y / number of random variables in Y
    :param n_dims: sum of X and Y dimensions
    :param means_std: std. dev. when sampling the kernel means
    """

    """  set parameters, calculate weights, means and covariances """
    self.n_kernels = n_kernels
    self.ndim = ndim_x + ndim_y
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.means_std = means_std
    self.weights = self._sample_weights(n_kernels) #shape(n_kernels,), sums to one
    self.means = np.random.normal(loc=np.zeros([self.ndim]), scale=self.means_std, size=[n_kernels, self.ndim]) #shape(n_kernels, n_dims)


    """ Sample cov matrixes and assure that cov matrix is pos definite"""
    self.covariances_x = helpers.project_to_pos_semi_def(np.abs(np.random.normal(loc=1, scale=0.5, size=(n_kernels, self.ndim_x, self.ndim_x)))) #shape(n_kernels, ndim_x, ndim_y)
    self.covariances_y = helpers.project_to_pos_semi_def(np.abs(np.random.normal(loc=1, scale=0.5, size=(n_kernels, self.ndim_y, self.ndim_y))))  # shape(n_kernels, ndim_x, ndim_y)

    """ some eigenvalues of the sampled covariance matrices can be exactly zero -> map to positive
    semi-definite subspace  """
    self.covariances = np.zeros(shape=(n_kernels, self.ndim, self.ndim))
    self.covariances[:, :ndim_x, :ndim_x] = self.covariances_x
    self.covariances[:, ndim_x:, ndim_x:] = self.covariances_y


    """ after mapping, define the remaining variables and collect frozen multivariate variables
      (x,y), x and y for later conditional draws """
    self.means_x = self.means[:, :ndim_x]
    self.means_y = self.means[:, ndim_x:]


    self.gaussians, self.gaussians_x, self.gaussians_y = [], [], []
    for i in range(n_kernels):
      self.gaussians.append(stats.multivariate_normal(mean=self.means[i,], cov=self.covariances[i]))
      self.gaussians_x.append(stats.multivariate_normal(mean=self.means_x[i,], cov=self.covariances_x[i]))
      self.gaussians_y.append(stats.multivariate_normal(mean=self.means_y[i,], cov=self.covariances_y[i]))

  def cdf(self, X, Y):
    """
       Determines the conditional cumulative probability density function P(Y<y|X=x).
       See "Conditional Gaussian Mixture Models for Environmental Risk Mapping" [Gilardi, Bengio] for the math.
       :param X: the position/conditional variable for the distribution P(Y<y|X=x), array_like, shape:(n_samples,
       ndim_x)
       :param Y: the on X conditioned variable Y, array_like, shape:(n_samples, ndim_y)
       :return: the cond. cumulative distribution of Y given X, for the given realizations of X with shape:(n_samples,)
       """
    X, Y = self._handle_input_dimensionality(X, Y)

    P_y = np.stack([self.gaussians_y[i].cdf(Y) for i in range(self.n_kernels)],
                   axis=1)  # shape(X.shape[0], n_kernels)
    W_x = self._W_x(X)
    print(W_x.sum(axis=1))

    cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)
    assert cond_prob.shape[0] == X.shape[0]
    return cond_prob

  def pdf(self, X, Y):
    """
    Determines the conditional probability density function P(Y|X).
    See "Conditional Gaussian Mixture Models for Environmental Risk Mapping" [Gilardi, Bengio] for the math.
    :param X: the position/conditional variable for the distribution P(Y|X), array_like, shape:(n_samples,
    ndim_x)
    :param Y: the on X conditioned variable Y, array_like, shape:(n_samples, ndim_y)
    :return: the cond. distribution of Y given X, for the given realizations of X with shape:(n_samples,)
    """
    X, Y = self._handle_input_dimensionality(X,Y)

    P_y = np.stack([self.gaussians_y[i].pdf(Y) for i in range(self.n_kernels)], axis=1) #shape(X.shape[0], n_kernels)
    W_x = self._W_x(X)

    cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)
    assert cond_prob.shape[0] == X.shape[0]
    return cond_prob

  def joint_pdf(self, X, Y):
    """
       Determines the joint probability density function P(X, Y).
       :param X: variable X for the distribution P(X, Y), array_like, shape:(n_samples, ndim_x)
       :param Y: variable Y for the distribution P(X, Y) array_like, shape:(n_samples, ndim_y)
       :return: the joint distribution of X and Y wih shape:(n_samples,)
       """
    X, Y = self._handle_input_dimensionality(X,Y)
    XY = np.concatenate([X,Y], axis=1)
    a = [self.weights[i] * self.gaussians[i].pdf(XY) for i in range(self.n_kernels)]
    p_i = np.stack(a, axis=1)
    return np.sum(p_i, axis=1)

  def simulate_conditional(self, X):
    """
    Draws for each x in X a sample y from P(y|x)
    :param X: array_like, shape:(n_samples, ndim_x)
    :return: X, Y  - Y is the arra
    """
    W_x = self._W_x(X)
    Y = np.zeros(shape=(X.shape[0], self.ndim_y))
    for i in range(X.shape[0]):
      discrete_dist = stats.rv_discrete(values=(range(self.n_kernels), W_x[i,:]))
      idx = discrete_dist.rvs()
      Y[i, :] = self.gaussians_y[idx].rvs()
    assert X.shape[0] == Y.shape[0]
    return X, Y

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

  def _sample_weights(self, n_weights):
    """
    samples density weights -> sum up to one
    :param n_weights: number of weights
    :return: ndarray of weights with shape (n_weights,)
    """
    weights = np.random.uniform(0, 1, size=[n_weights])
    return weights / np.sum(weights)

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

  def __str__(self):
    return str("\nProbabilistic model type: {}\nn_kernels: {}\nn_dim_x: {}\nn_dim_y: {}\nmeans_std: {}\n".format(self.__class__.__name__,
                                                                                                self.n_kernels, self.ndim_x, self.ndim_y, self.means))


if __name__=="__main__":
  model = GaussianMixture()
  model.plot(mode="joint_pdf")
