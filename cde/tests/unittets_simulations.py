import unittest
import warnings
import scipy.stats as stats
import numpy as np
from cde.density_simulation import *


class TestArmaJump(unittest.TestCase):

  def test_cdf_sample_consistency(self):
    np.random.seed(8787)
    arj = ArmaJump()

    x_cond = np.asarray([0.1 for _ in range(200000)])
    y_sample = arj.simulate_conditional(x_cond)


    cdf_callable = lambda y: arj.cdf(x_cond, y)
    _, p_val = stats.kstest(y_sample, cdf_callable)
    print("P-Val Kolmogorov:", p_val)

    self.assertGreaterEqual(p_val, 0.5)

  def test_skewness(self):
    np.random.seed(22)
    arj = ArmaJump(jump_prob=0.01)
    x_cond = np.asarray([0.1 for _ in range(200000)])
    y_sample = arj.simulate_conditional(x_cond)

    skew1 = stats.skew(y_sample)

    arj = ArmaJump(jump_prob=0.1)
    y_sample = arj.simulate_conditional(x_cond)

    skew2 = stats.skew(y_sample)

    self.assertLessEqual(skew2, skew1)

  def test_mean(self):
    np.random.seed(22)
    arj = ArmaJump(c=0.1, jump_prob=0.00)
    x_cond = np.asarray([0.1])
    mean = arj.mean_(x_cond)
    self.assertAlmostEqual(mean, 0.1)

    arj = ArmaJump(c=0.1, jump_prob=0.1)
    mean = arj.mean_(x_cond)
    self.assertLessEqual(mean, 0.1)

  def test_cov(self):
    np.random.seed(22)
    arj = ArmaJump(c=0.1, jump_prob=0.00, std=0.1)
    x_cond = np.asarray([0.1])
    cov = arj.covariance(x_cond)[0][0][0]
    self.assertAlmostEqual(cov, 0.1**2)

    arj = ArmaJump(c=0.1, jump_prob=0.1, std=0.1)
    cov = arj.covariance(x_cond)[0][0][0]
    self.assertGreater(cov, 0.1 ** 2)
