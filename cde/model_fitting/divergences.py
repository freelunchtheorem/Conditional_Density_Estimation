import numpy as np
from cde.BaseConditionalDensity import ConditionalDensity
from cde.utils.integration import mc_integration_student_t

_FUN_KL = lambda p, q: p * np.log(p / q)
_FUN_JS = lambda p, q: 0.5 * p * np.log(p / q) + 0.5 * q * np.log(q / p)
_FUN_HELLINGER_2 = lambda p, q: (np.sqrt(p) - np.sqrt(q))**2

def kl_divergence_pdf(p, q, x_cond, n_samples=10 ** 5):
  """ Computes the Kullback–Leibler divergence KL[p ; q] via monte carlo integration
  using importance sampling with a student-t proposal distribution

  Args:
   p: conditional distribution object p(y|x)
   q: conditional distribution object q(y|x)
   x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
   n_samples: number of samples for monte carlo integration over the y space

  Returns:
    KL divergence of each x value to condition on - numpy array of shape (n_values,)
  """
  return _divergence_mc(p, q, x_cond, _FUN_KL, n_samples)

def js_divergence_pdf(p, q, x_cond, n_samples=10 ** 5):
  """ Computes the Jensen-Shannon divergence JS[p ; q] via monte carlo integration
  using importance sampling with a student-t proposal distribution

  Args:
   p: conditional distribution object p(y|x)
   q: conditional distribution object q(y|x)
   x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
   n_samples: number of samples for monte carlo integration over the y space

  Returns:
    JS divergence of each x value to condition on - numpy array of shape (n_values,)
  """
  divergence_fun = lambda p, q: 0.5 * p * np.log(p / q) + 0.5 * q * np.log(q / p)
  return _divergence_mc(p, q, x_cond, divergence_fun, n_samples)

def hellinger_distance_pdf(p, q, x_cond, n_samples=10 ** 5):
  """ Computes the Hellinger Distance H[p ; q] via monte carlo integration
  using importance sampling with a student-t proposal distribution

  Args:
   p: conditional distribution object p(y|x)
   q: conditional distribution object q(y|x)
   x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
   n_samples: number of samples for monte carlo integration over the y space

  Returns:
    Hellinger distance for each x value to condition on - numpy array of shape (n_values,)
  """
  hellinger_squared = _divergence_mc(p, q, x_cond, _FUN_HELLINGER_2, n_samples)
  return np.sqrt(0.5 * hellinger_squared)

def divergence_measures_pdf(p, q, x_cond, n_samples=10**5):
  """ Computes the
      - Hellinger Distance H[p ; q]
      - Kullback–Leibler divergence KL[p ; q]
      - Jennsen-Shannon divergence JS[p ; q]
      via monte carlo integration using importance sampling with a student-t proposal distribution

    Args:
     p: conditional distribution object p(y|x)
     q: conditional distribution object q(y|x)
     x_cond: x values to condition on - numpy array of shape (n_values, ndim_x)
     n_samples: number of samples for monte carlo integration over the y space

    Returns:
      (hellinger_dists, kl_divs, js_divs) - tuple of numpy arrays of shape (n_values,)
    """
  fun_div_measures_stack = lambda p, q: np.stack([_FUN_HELLINGER_2(p,q), _FUN_KL(p,q), _FUN_JS(p,q)], axis=1) # np.sqrt(_FUN_HELLINGER_2(p,q))
  div_measure_stack = _divergence_mc(p, q, x_cond, fun_div_measures_stack, n_samples, n_measures=3)
  assert div_measure_stack.shape == (x_cond.shape[0], 3)
  h_divs, kl_divs, js_divs = div_measure_stack[:, 0], div_measure_stack[:, 1], div_measure_stack[:, 2]
  return np.sqrt(0.5 * h_divs), kl_divs, js_divs

def _divergence_mc(p, q, x_cond, divergenc_fun, n_samples=10 ** 5, n_measures=1):
  assert x_cond.ndim == 2 and x_cond.shape[1] == q.ndim_x

  P = p.pdf
  Q = q.pdf

  def _div(x_tiled, y_samples):
    p = P(x_tiled, y_samples).flatten()
    q = Q(x_tiled, y_samples).flatten()
    q = np.ma.masked_where(q < 10 ** -64, q).flatten()
    p = np.ma.masked_where(p < 10 ** -64, p).flatten()

    r = divergenc_fun(p, q)
    return r.filled(0)

  if n_measures == 1:
    distances = np.zeros(x_cond.shape[0])
  else:
    distances = np.zeros((x_cond.shape[0], n_measures))
  mu_proposal, std_proposal = p._determine_mc_proposal_dist()
  for i in range(x_cond.shape[0]):
    x = np.tile(x_cond[i].reshape((1, x_cond[i].shape[0])), (n_samples, 1))
    func = lambda y: _make_2d(_div(x, y))
    distances[i] = mc_integration_student_t(func, q.ndim_y, n_samples=n_samples, loc_proposal=mu_proposal, scale_proposal=std_proposal)
  assert distances.shape[0] == x_cond.shape[0]
  return distances


""" helpers """

def _make_2d(a):
  return np.reshape(a, (a.shape[0], -1))