import copy
import warnings
from math import exp as _exp

import numpy as np
import numbers

import scipy.integrate as integrate

from cde.utils.distribution import multidim_t_pdf, multidim_t_rvs, multivariate_t_rvs

N_SAMPLES_ADAPT = 10**3

def numeric_integation(func, n_samples=10 ** 5, bound_lower=-10**3, bound_upper=10**3):
  """ Numeric integration over one dimension using the trapezoidal rule

     Args:
       func: function to integrate over - must take numpy arrays of shape (n_samples,) as first argument
             and return a numpy array of shape (n_samples,)
       n_samples: (int) number of samples

     Returns:
       approximated integral - numpy array of shape (ndim_out,)
    """
  # proposal distribution
  y_samples = np.squeeze(np.linspace(bound_lower, bound_upper, num=n_samples))
  values = func(y_samples)
  integral = integrate.trapz(values, y_samples)
  return integral


def mc_integration_student_t(func, ndim, n_samples=10 ** 6, batch_size=None, loc_proposal=0,
                             scale_proposal=2, dof=6):
    """ Monte carlo integration using importance sampling with a cauchy distribution

    Args:
      func: function to integrate over - must take numpy arrays of shape (n_samples, ndim) as first argument
            and return a numpy array of shape (n_samples, ndim_out)
      ndim: (int) number of dimensions to integrate over
      n_samples: (int) number of samples
      batch_size: (int) batch_size for junking the n_samples in batches (optional)

    Returns:
      approximated integral - numpy array of shape (ndim_out,)

    """
    if batch_size is None:
        n_batches = 1
        batch_size = n_samples
    else:
        n_batches = n_samples // batch_size + int(n_samples % batch_size > 0)

    batch_results = []

    if isinstance(loc_proposal, numbers.Number):
        loc_proposal = np.ones(ndim) * loc_proposal
    if isinstance(scale_proposal, numbers.Number):
        scale_proposal = np.ones(ndim) * scale_proposal


    for j in range(n_batches):
        samples = multidim_t_rvs(loc_proposal, scale_proposal, dof=dof, N=batch_size)
        f = np.expand_dims(multidim_t_pdf(samples, loc_proposal, scale_proposal, dof), axis=1)
        r = func(samples)
        assert r.ndim == 2, 'func must return a 2-dimensional numpy array'
        f = np.tile(f, (1, r.shape[1]))  # bring f into same shape like r
        assert (f.shape == r.shape)
        batch_results.append(np.mean(r / f, axis=0))

    result = np.mean(np.stack(batch_results, axis=0), axis=0)
    return result

""" Other helpers """
import os, sys

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()