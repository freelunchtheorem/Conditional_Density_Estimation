import numpy as np
import scipy.stats as stats
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation
from cde.utils.misc import project_to_pos_semi_def
from sklearn.mixture import GaussianMixture as GMM

class GaussianMixture(BaseConditionalDensitySimulation):
  """
  This model allows to fit and sample from a uni- bi- or multivariate Gaussian mixture model with diagonal covariance matrices.
  The mixture model is composed by a linear combination of an arbitrary number of components n_kernels. Means, covariances and weights are
  estimated by Maximum-Likelihood for each component. It is possible to specify the number of kernels to define the modality of the
  distribution and also dimensionality for both x and y. The component means are initialized randomly according to given standard
  deviation. Also the weights are initialized randomly.

  Args:
    n_kernels: number of mixture components
    ndim_x: dimensionality of X / number of random variables in X
    ndim_y: dimensionality of Y / number of random variables in Y
    means_std: std. dev. when sampling the kernel means
    random_seed: seed for the random_number generator
  """

  def __init__(self, n_kernels=5, ndim_x=1, ndim_y=1, means_std=1.5, random_seed=None):

    self.random_state = np.random.RandomState(seed=random_seed) # random state for sampling data
    self.random_state_params = np.random.RandomState(seed=20) # fixed random state for sampling GMM params
    self.random_seed = random_seed

    self.has_pdf = True
    self.has_cdf = True
    self.can_sample = True

    """  set parameters, calculate weights, means and covariances """
    self.n_kernels = n_kernels
    self.ndim = ndim_x + ndim_y
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.means_std = means_std
    self.weights = self._sample_weights(n_kernels) #shape(n_kernels,), sums to one
    self.means = self.random_state_params.normal(loc=np.zeros([self.ndim]), scale=self.means_std, size=[n_kernels, self.ndim]) #shape(n_kernels, n_dims)


    """ Sample cov matrixes and assure that cov matrix is pos definite"""
    self.covariances_x = project_to_pos_semi_def(np.abs(self.random_state_params.normal(loc=1, scale=0.5, size=(n_kernels, self.ndim_x, self.ndim_x)))) #shape(n_kernels, ndim_x, ndim_y)
    self.covariances_y = project_to_pos_semi_def(np.abs(self.random_state_params.normal(loc=1, scale=0.5, size=(n_kernels, self.ndim_y, self.ndim_y))))  # shape(n_kernels, ndim_x, ndim_y)

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

    # approximate data statistics
    self.y_mean, self.y_std = self._compute_data_statistics()

  def pdf(self, X, Y):
    """ conditional probability density function P(Y|X)
        See "Conditional Gaussian Mixture Models for Environmental Risk Mapping" [Gilardi, Bengio] for the math.

    Args:
      X: the position/conditional variable for the distribution P(Y|X), array_like, shape:(n_samples, ndim_x)
      Y: the on X conditioned variable Y, array_like, shape:(n_samples, ndim_y)

    Returns:
      the cond. distribution of Y given X, for the given realizations of X with shape:(n_samples,)
    """

    X, Y = self._handle_input_dimensionality(X,Y)

    P_y = np.stack([self.gaussians_y[i].pdf(Y) for i in range(self.n_kernels)], axis=1) #shape(X.shape[0], n_kernels)
    W_x = self._W_x(X)

    cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)
    assert cond_prob.shape[0] == X.shape[0]
    return cond_prob

  def cdf(self, X, Y):
    """ conditional cumulative probability density function P(Y<y|X=x).
       See "Conditional Gaussian Mixture Models for Environmental Risk Mapping" [Gilardi, Bengio] for the math.

    Args:
      X: the position/conditional variable for the distribution P(Y<y|X=x), array_like, shape:(n_samples, ndim_x)
      Y: the on X conditioned variable Y, array_like, shape:(n_samples, ndim_y)

    Returns:
      the cond. cumulative distribution of Y given X, for the given realizations of X with shape:(n_samples,)
    """

    X, Y = self._handle_input_dimensionality(X, Y)

    P_y = np.stack([self.gaussians_y[i].cdf(Y) for i in range(self.n_kernels)],
                   axis=1)  # shape(X.shape[0], n_kernels)
    W_x = self._W_x(X)

    cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)
    assert cond_prob.shape[0] == X.shape[0]
    return cond_prob

  def joint_pdf(self, X, Y):
    """ joint probability density function P(X, Y)

    Args:
      X: variable X for the distribution P(X, Y), array_like, shape:(n_samples, ndim_x)
      Y: variable Y for the distribution P(X, Y) array_like, shape:(n_samples, ndim_y)

    Returns:
      the joint distribution of X and Y wih shape:(n_samples,)
    """

    X, Y = self._handle_input_dimensionality(X,Y)
    XY = np.concatenate([X,Y], axis=1)
    a = [self.weights[i] * self.gaussians[i].pdf(XY) for i in range(self.n_kernels)]
    p_i = np.stack(a, axis=1)
    return np.sum(p_i, axis=1)

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """

    X = self._handle_input_dimensionality(X)

    if np.all(np.all(X == X[0, :], axis=1)):
      return self._simulate_cond_rows_same(X)
    else:
      return self._simulate_cond_rows_individually(X)


  def simulate(self, n_samples=1000):
    """ Draws random samples from the unconditional distribution p(x,y)

    Args:
      n_samples: (int) number of samples to be drawn from the conditional distribution

    Returns:
      (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """

    assert n_samples > 0

    n_samples_comp = self.random_state.multinomial(n_samples, self.weights)

    samples = np.vstack([gaussian.rvs(size=n, random_state=self.random_state)
                                     for gaussian, n in zip(self.gaussians, n_samples_comp)])

    # shuffle rows to make data i.i.d.
    self.random_state.shuffle(samples)

    x_samples = samples[:, :self.ndim_x]
    y_samples = samples[:, self.ndim_x:]

    assert x_samples.shape == (n_samples, self.ndim_x)
    assert y_samples.shape == (n_samples, self.ndim_y)
    return x_samples, y_samples

  def mean_(self, x_cond, n_samples=None):
    """ Conditional mean of the distribution
     Args:
       x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

     Returns:
       Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
     """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x

    W_x = self._W_x(x_cond)
    means = W_x.dot(self.means_y)
    return means

  def covariance(self, x_cond, n_samples=None):
    """ Covariance of the distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x
    W_x = self._W_x(x_cond)

    covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))

    glob_mean = self.mean_(x_cond)

    for i in range(x_cond.shape[0]):
      c1 = np.zeros((self.ndim_y, self.ndim_y))
      c2 = np.zeros(c1.shape)
      weights = W_x[i]
      for j in range(weights.shape[0]):
        c1 += weights[j] * self.covariances_y[j]
        a = (self.means_y[j] - glob_mean[i])
        d = weights[j] * np.outer(a, a)
        c2 += d
      covs[i] = c1 + c2
    return covs

  def _simulate_cond_rows_individually(self, X):
    W_x = self._W_x(X)
    y_samples = np.zeros(shape=(X.shape[0], self.ndim_y))

    for i in range(X.shape[0]):
      discrete_dist = stats.rv_discrete(values=(range(self.n_kernels), W_x[i, :]))
      idx = discrete_dist.rvs(random_state=self.random_state)
      y_samples[i, :] = self.gaussians_y[idx].rvs(random_state=self.random_state)

    return X, y_samples

  def _simulate_cond_rows_same(self, X):
    n_samples = X.shape[0]
    weights = self._W_x(X)[0]

    n_samples_comp = self.random_state.multinomial(n_samples, weights)

    y_samples = np.vstack([gaussian.rvs(size=n, random_state=self.random_state)
                         for gaussian, n in zip(self.gaussians_y, n_samples_comp)])

    # shuffle rows to make data i.i.d.
    self.random_state.shuffle(y_samples)

    return X, y_samples


  def _sample_weights(self, n_weights):
    """ samples density weights -> sum up to one
    Args:
      n_weights: number of weights
    Returns:
      ndarray of weights with shape (n_weights,)
    """
    weights = self.random_state_params.uniform(0, 1, size=[n_weights])
    return weights / np.sum(weights)

  def _W_x(self, X):
    """ Helper function to normalize the joint density P(Y,X) by the marginal density P(X)

    Args:
      X: conditional random variable, array_like, shape:(n_samples, ndim_x)

    Return:
      the normalized weighted marginal gaussian distributions P(X) for each n_kernel, shape:(n_samples,n_kernels)
    """
    assert X.ndim == 2 and X.shape[1] == self.ndim_x
    if X.shape[0] == 1:
      w_p = np.stack([np.array([self.weights[i] * self.gaussians_x[i].pdf(X)]) for i in range(self.n_kernels)], axis=1)
    else:
      w_p = np.stack([self.weights[i] * self.gaussians_x[i].pdf(X) for i in range(self.n_kernels)], axis=1)
    normalizing_term = np.sum(w_p, axis=1)
    result = w_p / normalizing_term[:,None]
    return result

  def __str__(self):
    return "\nProbabilistic model type: {}\nn_kernels: {}\nn_dim_x: {}\nn_dim_y: {}\nmeans_std: {}\n".format(self.__class__.__name__,
                                                                    self.n_kernels, self.ndim_x, self.ndim_y, self.means)

  def __unicode__(self):
    return self.__str__()

