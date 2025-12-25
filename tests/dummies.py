import json
import sys
import os
import time

_DEBUG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".debug_logs"))
_DEBUG_LOG_PATH = os.path.join(_DEBUG_DIR, "debug.log")

def _append_debug_log(log_entry):
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    with open(_DEBUG_LOG_PATH, "a") as _f:
        _f.write(json.dumps(log_entry) + "\n")

def _log_event(hypothesis_id, location, message, data):
    entry = {
        "sessionId": "debug-session",
        "runId": "prefix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    _append_debug_log(entry)


# region agent log
try:
    _log_event(
        "H2",
        "tests/dummies.py:pre_import",
        "sys.path snapshot before imports",
        {"sys_path": sys.path[:5]},
    )
except Exception:
    pass
# endregion

try:
    # region agent log
    import scipy.stats as stats
    _log_event(
        "H1",
        "tests/dummies.py:scipy_import",
        "scipy imported successfully",
        {"scipy_file": getattr(stats, "__file__", None)},
    )
    # endregion
except Exception as e:
    # region agent log
    _log_event(
        "H3",
        "tests/dummies.py:scipy_import",
        "scipy import failed",
        {"error": str(e)},
    )
    # endregion
    raise

try:
    # region agent log
    import numpy as np
    _log_event(
        "H1",
        "tests/dummies.py:numpy_import",
        "numpy imported successfully",
        {"numpy_file": getattr(np, "__file__", None), "numpy_version": np.__version__},
    )
    # endregion
except Exception as e:
    # region agent log
    _log_event(
        "H3",
        "tests/dummies.py:numpy_import",
        "numpy import failed",
        {"error": str(e)},
    )
    # endregion
    raise

try:
    # region agent log
    multiarray_path = getattr(getattr(np, "core", None), "multiarray", None)
    multiarray_file = getattr(multiarray_path, "__file__", None)
    _log_event(
        "H4",
        "tests/dummies.py:numpy_multiarray",
        "numpy core multiarray path",
        {
            "multiarray_file": multiarray_file,
            "sys_modules_numpy": str(sys.modules.get("numpy")),
            "sys_modules_multiarray": str(sys.modules.get("numpy._core.multiarray")),
        },
    )
    # endregion
except Exception as e:
    # region agent log
    _log_event(
        "H4",
        "tests/dummies.py:numpy_multiarray",
        "failed to log numpy multiarray info",
        {"error": str(e)},
    )
    # endregion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.density_simulation import BaseConditionalDensitySimulation
from cde.density_estimator import BaseDensityEstimator



class GaussianDummy(BaseDensityEstimator):

  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = mean
    # check if mean is scalar
    if isinstance(self.mean, list):
      self.mean = np.array(self.ndim_y * [self.mean])

    self.cov = cov
    if self.cov is None:
      self.cov = np.identity(self.ndim_y)
    assert self.cov.shape[0] == self.cov.shape[1] == self.ndim_y

    # safe data stats
    self.y_mean = self.mean
    self.y_std = np.sqrt(np.diagonal(self.cov))

    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

  def fit(self, X, Y, verbose=False):
    self.fitted = True

  def pdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.gaussian.cdf(Y)

  def sample(self, X):
    X = self._handle_input_dimensionality(X)
    if np.size(X) == 1:
      Y = self.gaussian.rvs(size=1)
    else:
      Y = self.gaussian.rvs(size=(X.shape[0]))
    return X,Y

  def __str__(self):
    return str('\nEstimator type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n' .format(self.__class__.__name__, self.ndim_x, self.ndim_y, self.mean))


class SkewNormalDummy(BaseDensityEstimator):

  def __init__(self, shape=1, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    assert ndim_y == 1, "only on-dimensional y supported for skew normal dummy"

    self.shape = shape

    self.distribution = stats.skewnorm(a=shape)
    self.fitted = False

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

  def fit(self, X, Y, verbose=False):
    self.fitted = True

  def pdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.distribution.pdf(Y).flatten()

  def cdf(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y)
    return self.distribution.cdf(Y).flatten()

  def sample(self, X):
    X = self._handle_input_dimensionality(X)
    if np.size(X) == 1:
      Y = self.distribution.rvs(size=1)
    else:
      Y = self.distribution.rvs(size=(X.shape[0]))
    return X,Y

  @property
  def skewness(self):
    gamma = self.shape / np.sqrt(1+self.shape**2)
    skew = ((4-np.pi) / 2) * ((gamma * np.sqrt(2/np.pi))**3 / (1 - 2 * gamma**2 / np.pi )**(3/2))
    return skew

  @property
  def kurtosis(self):
    gamma = self.shape / np.sqrt(1 + self.shape ** 2)
    kurt = 2*(np.pi - 3) * (gamma * np.sqrt(2/np.pi))**4 / (1 - 2*gamma**2 / np.pi)**2
    return kurt


class SimulationDummy(BaseConditionalDensitySimulation):
  def __init__(self, mean=2, cov=None, ndim_x=1, ndim_y=1, has_cdf=True, has_pdf=True, can_sample=True):
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.ndim = self.ndim_x + self.ndim_y

    self.mean = mean
    self.cov = cov
    # check if mean is scalar
    if isinstance(self.mean, list):
      self.mean = np.array(self.ndim_y*[self.mean])

    if self.cov is None:
      self.cov = np.identity(self.ndim_y)

    self.gaussian = stats.multivariate_normal(mean=self.mean, cov=self.cov)
    self.fitted = False

    self.can_sample = can_sample
    self.has_pdf = has_pdf
    self.has_cdf = has_cdf

  def pdf(self, X, Y):
    return self.gaussian.pdf(Y)

  def cdf(self, X, Y):
    return self.gaussian.cdf(Y)

  def simulate(self, n_samples=1000):
    assert n_samples > 0
    X = self.gaussian.rvs(size=n_samples)
    Y = self.gaussian.rvs(size=n_samples)
    return X, Y

  def simulate_conditional(self, X):
    Y = self.gaussian.rvs(size=X.shape[0])
    return X, Y

  def __str__(self):
    return str('\nProbabilistic model type: {}\n n_dim_x: {}\n n_dim_y: {}\n mean: {}\n cov: {}\n'.format(self.__class__.__name__, self.ndim_x,
                self.ndim_y, self.mean, self.cov))
