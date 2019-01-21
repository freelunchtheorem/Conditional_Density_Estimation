
import numpy as np
import math
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation

class JumpDiffusionModel(BaseConditionalDensitySimulation):
  """
  Jump-Diffustion continous time model by Christoffersen et al. (2016), "Time-varying Crash Risk: The Role of Market Liquiditiy"

  Args:
    random_seed: seed for the random_number generator
  """

  def __init__(self, random_seed=None):
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    # Parameters based on the paper with slight modifications
    self.r = 0.0
    self.kappa_V = 3.011
    self.theta_V = 0.0365
    self.xi_V = 0.346
    self.kappa_L = 2.353
    self.theta_L = 0.171
    self.xi_L = 0.158
    self.kappa_psi = 0.662
    self.theta_psi = 0.101
    self.xi_psi = 0.204
    self.rho = -0.353
    self.theta = -0.037
    self.delta = 0.031
    # gamma = 0.118
    # gamma_V = 18.38
    # gamma_L = 9.259
    self.gamma = 0.4
    self.gamma_V = 90
    self.gamma_L = 25

    """ Starting values for the model variables (unconditional expectation except for log-return) """

    self.y_0 = 0
    self.V_0 = 0.0365
    self.L_0 = 0.171
    self.Psi_0 = 0.101

    self.ndim_x = 3
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    # approximate data statistics
    self.y_mean, self.y_std = self._compute_data_statistics()

    self.has_cdf = False
    self.has_pdf = False
    self.can_sample = True

  def pdf(self, X, Y):
    raise NotImplementedError

  def cdf(self, X, Y):
    raise NotImplementedError

  def joint_pdf(self, X, Y):
    raise NotImplementedError

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

     Args:
       X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, 3)
          thereby x is a horizontal stack of V, L and Psi
          -> x = (V, L, Psi)

     Returns: (X,Y)
       - X: the x to of the conditional samples (identical with argument X)
       - Y: Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, 1)

    """
    X = self._handle_input_dimensionality(X)
    V_sim, L_sim, Psi_sim = X[:,0], X[:,1], X[:,2]

    Y, _, _, _ = self._simulate_one_step(V_sim, L_sim, Psi_sim)
    Y = np.expand_dims(Y, axis=1)
    assert Y.shape == (X.shape[0], self.ndim_y)
    return X, Y

  def simulate(self, n_samples=10000):
    """ Simulates a time-series of n_samples time steps

     Args:
       samples: (int) number of samples to be drawn from the joint distribution P(X,Y)

     Returns: (X,Y)
       - X: horizontal stack of simulated V (spot vol), L (illigudity) and Psi (latent state) - numpy array of shape (n_samples, 3)
       - Y: log returns drawn from P(Y|X) - numpy array of shape (n_samples, 1)
    """

    assert n_samples > 0
    N = 1

    y_sim = np.zeros((n_samples+1, N))
    V_sim = np.zeros((n_samples+1, N))
    L_sim = np.zeros((n_samples+1, N))
    Psi_sim = np.zeros((n_samples+1, N))

    y_sim[0, :] = np.full((N,), self.y_0)
    V_sim[0, :] = np.full((N,), self.V_0)
    L_sim[0, :] = np.full((N,), self.L_0)
    Psi_sim[0, :] = np.full((N,), self.Psi_0)

    for i in range(0, n_samples):
      y_sim[i+1], V_sim[i+1], L_sim[i+1], Psi_sim[i+1] = self._simulate_one_step(V_sim[i], L_sim[i], Psi_sim[i])

    X = np.hstack([V_sim[:n_samples], L_sim[:n_samples], Psi_sim[:n_samples]])
    Y = y_sim[1:]
    assert Y.shape == (n_samples,self.ndim_y) and X.shape == (n_samples,self.ndim_x)
    return X, Y

  def _simulate_one_step(self, V_sim, L_sim, Psi_sim):
    assert V_sim.ndim == L_sim.ndim == Psi_sim.ndim
    assert V_sim.shape[0] == L_sim.shape[0] == Psi_sim.shape[0]

    N = V_sim.shape[0]
    y_sim = np.full((N,), 0)

    xi = math.exp(self.theta + (self.delta ** 2) / 2) - 1
    dt = 1 / 252
    lambda_t = Psi_sim + self.gamma_V * V_sim + self.gamma_L * L_sim

    Psi_sim = np.maximum(0, Psi_sim + self.kappa_psi * (self.theta_psi - Psi_sim) * dt + self.xi_psi * (
              (Psi_sim * dt) ** 0.5) * self.random_state.normal(size=(N,)))
    L_shocks = self.random_state.normal(size=(N,))
    L_sim = np.maximum(0, L_sim + self.kappa_L * (self.theta_L - L_sim) * dt + self.xi_L * ((L_sim * dt) ** 0.5) * L_shocks)
    V_shocks = self.random_state.normal(size=(N,))
    V_sim = np.maximum(0,
                       V_sim + self.kappa_V * (self.theta_V - V_sim) * dt + self.gamma * self.kappa_L * (self.theta_L - L_sim) * dt + self.xi_V * (
                                 (V_sim * dt) ** 0.5) * V_shocks + self.gamma * self.xi_L * ((L_sim * dt) ** 0.5) * L_shocks)

    q = self.random_state.normal(loc=self.theta, scale=self.delta, size=(N,))
    jumps = self.random_state.poisson(lam=lambda_t * dt, size=(1, N)).T[:, 0]
    y_shocks = self.random_state.normal(size=(N,))
    y_sim = y_sim + (self.r - 0.5 * V_sim - xi * lambda_t + 1.554 * (V_sim ** 0.5)) * dt + ((V_sim * dt) ** 0.5) * (
              ((1 - self.rho ** 2) ** 0.5) * y_shocks + self.rho * V_shocks) + q * jumps

    return y_sim, V_sim, L_sim, Psi_sim

  def __str__(self):
    return "\nProbabilistic model type: {}\n parameters: {{}}".format(self.__class__.__name__, **self.__dict__)

  def __unicode__(self):
    return self.__str__()