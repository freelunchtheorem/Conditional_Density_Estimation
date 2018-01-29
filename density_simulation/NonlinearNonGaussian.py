import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from density_simulation import ConditionalDensity

class NonlinearNonGaussian(ConditionalDensity):
  """
  Model by Christoffersen et al. (2016), "Time-varying Crash Risk: The Role of Market Liquiditiy"
  """

  def __init__(self, std=1):
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

    self.x_0 = 0
    self.V_0 = 0.0365
    self.L_0 = 0.171
    self.Psi_0 = 0.101

    """ number of days over which the model is simulated. Discretization is at the daily level. """
    self.timesteps = 1

    self.ndim_x = 1
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

  def pdf(self, X, Y):
    raise NotImplementedError
  def cdf(self, X, Y):
    raise NotImplementedError

  def joint_pdf(self, X, Y):
    raise NotImplementedError

  def simulate_conditional(self, X):
    return NotImplementedError

  def simulate(self, n_samples=1000):
    """
    Draws (n_samples) instances from the
    (Implementation by Simon Walther)
    :param x, V, L, Psi: Starting values for the model variables.
    :param n_samples: Number of simulations
    """
    assert n_samples > 0

    x = np.zeros((n_samples,))
    x[0] = self.x_0
    V = np.zeros((n_samples,))
    V[0] = self.V_0
    L = np.zeros((n_samples,))
    L[0] = self.L_0
    Psi = np.zeros((n_samples,))
    Psi[0] = self.Psi_0


    x_sim = np.full((n_samples,), x)
    V_sim = np.full((n_samples,), V)
    L_sim = np.full((n_samples,), L)
    Psi_sim = np.full((n_samples,), Psi)

    xi = math.exp(self.theta + (self.delta ** 2) / 2) - 1
    dt = 1 / 252
    for t in range(0, self.timesteps):
      lambda_t = Psi_sim + self.gamma_V * V_sim + self.gamma_L * L_sim

      Psi_sim = np.maximum(0, Psi_sim + self.kappa_psi * (self.theta_psi - Psi_sim) * dt + self.xi_psi * ((Psi_sim * dt) ** 0.5) * np.random.normal(
        size=(n_samples,)))
      L_shocks = np.random.normal(size=(n_samples,))
      L_sim = np.maximum(0, L_sim + self.kappa_L * (self.theta_L - L_sim) * dt + self.xi_L * ((L_sim * dt) ** 0.5) * L_shocks)
      V_shocks = np.random.normal(size=(n_samples,))
      V_sim = np.maximum(0, V_sim + self.kappa_V * (self.theta_V - V_sim) * dt + self.gamma * self.kappa_L * (self.theta_L - L_sim) * dt + self.xi_V
                         * ((V_sim * dt) ** 0.5) * V_shocks + self.gamma * self.xi_L * ((L_sim * dt) ** 0.5) * L_shocks)

      q = np.random.normal(loc=self.theta, scale=self.delta, size=(n_samples,))
      jumps = np.random.poisson(lam=lambda_t * dt, size=(1, n_samples)).T[:, 0]
      x_shocks = np.random.normal(size=(n_samples,))
      x_sim = x_sim + (self.r - 0.5 * V_sim - xi * lambda_t + 1.554 * (V_sim ** 0.5)) * dt + ((V_sim * dt) ** 0.5) * (
      ((1 - self.rho ** 2) ** 0.5) * x_shocks + self.rho * V_shocks) + q * jumps

    return x_sim, V_sim, L_sim, Psi_sim


  def __str__(self):
    return "\nProbabilistic model type: {}\n parameters: {{}}".format(self.__class__.__name__, **self.__dict__)

  def __unicode__(self):
    return self.__str__()