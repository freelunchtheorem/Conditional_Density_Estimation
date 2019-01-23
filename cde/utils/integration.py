import copy
import warnings
from math import exp as _exp

import numpy as np
import numbers

from pypmc.density.mixture import create_t_mixture
from pypmc.mix_adapt.pmc import student_t_pmc
from pypmc.tools import History as _History
import scipy.integrate as integrate

from cde.utils.distribution import _multidim_t_pdf, _multidim_t_rvs, _multivariate_t_rvs

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
  y_samples = np.linspace(bound_lower, bound_upper, num=n_samples)
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
        samples = _multidim_t_rvs(loc_proposal, scale_proposal, dof=dof, N=batch_size)
        f = np.expand_dims(_multidim_t_pdf(samples, loc_proposal, scale_proposal, dof), axis=1)
        r = func(samples)
        assert r.ndim == 2, 'func must return a 2-dimensional numpy array'
        f = np.tile(f, (1, r.shape[1]))  # bring f into same shape like r
        assert (f.shape == r.shape)
        batch_results.append(np.mean(r / f, axis=0))

    result = np.mean(np.stack(batch_results, axis=0), axis=0)
    return result

def mc_integration_adaptive(fun, log_prob, ndim, n_samples=10 ** 6, adaptive=True, n_em_steps=4,
                            n_comps=2, loc_proposal=0, scale_proposal=1, random_state=None):
    """
    Estimates the integral $\int f(x) * p(x) dx = E_p[f(x)]$ with (adaptive) importance sampling.
    The init_proposal distribution is a mixture of student-t distributions and is adapted with PMC
    (https://link.springer.com/article/10.1007/s11222-008-9059-)
    Args:
        fun: (callable) function that receives a numpy array and returns a numpy array with one dimension
        log_prob: (callable) log_probability log p(x)
        ndim: (int) number of dimensions of x
        n_samples: (int) number of samples
        adaptive: (bool) whether to use an adaptive init_proposal distribution
                    If true, the population monte carlo (PMC) algorithm is used to optimize the init_proposal dist
        n_em_steps: (int) Number of expectation maximization steps for the PMC
    Returns:
        (float) estimate of the integral
    """
    if random_state is None: random_state = np.random.RandomState(None)

    if isinstance(loc_proposal, numbers.Number):
        loc_proposal = np.ones(ndim) * loc_proposal
    if isinstance(scale_proposal, numbers.Number):
        scale_proposal = np.ones(ndim) * scale_proposal

    log_target_adaptation = lambda x: log_prob(x).flatten() + np.log(np.abs(fun(x).flatten()))

    # create initial init_proposal dist (mixture of student_t distributions)
    dofs = random_state.uniform(low=2.5, high=20.0, size=n_comps) # sample dof of initial components
    weights = random_state.uniform(low=0.0, high=1.0, size=n_comps)
    weights = weights / np.mean(weights)
    means = [loc_proposal + 0.5*scale_proposal * random_state.normal(loc=np.zeros(ndim), scale=1.0) for _ in range(n_comps)]
    covs = [np.diag(scale_proposal) for _ in range(n_comps)]
    proposal_dist = create_t_mixture(means, covs, dofs, weights)

    sampler = _ImportanceSamplerTMixture(log_target_adaptation, proposal_dist, rng=random_state)

    # adapt init_proposal distribution
    warnings.filterwarnings('error')
    if adaptive:
        with NoStdStreams():
            generating_components = []
            for i in range(n_em_steps):
                generating_components.append(sampler.run(N_SAMPLES_ADAPT, trace_sort=True))
                samples, weights = sampler.samples[-1], sampler.weights[-1].flatten()
                try:
                  proposal_dist = student_t_pmc(samples, sampler.proposal,
                                          weights,
                                          latent=generating_components[-1],
                                          mincount=20, rb=True, copy=True)
                except (RuntimeWarning, ValueError):
                  warnings.warn("Adaptive PMC failed - falling back on proposal dist")
                  break
    warnings.filterwarnings('default')

    # monte carlo integration with importance sampling from init_proposal distribution
    log_target = lambda x: log_prob(x).flatten()
    sampler = _ImportanceSamplerTMixture(log_target, proposal_dist, rng=random_state)
    sampler.run(N=n_samples)
    samples, weights = sampler.samples[-1], sampler.weights[-1]
    weights = np.tile(np.reshape(weights, (samples.shape[0], 1)), (1, samples.shape[1]))
    result = np.mean(np.multiply(weights, fun(samples)), axis=0)
    return result


""" Helpers for adaptive importance sampling """

class _Student_t_Mixture_Density_Sampler():
  def __init__(self, student_t_mixture):
    assert len(student_t_mixture.components)
    self.weights = student_t_mixture.weights
    self.comps = student_t_mixture.components

  def propose(self, N, rng=None, trace=False):
    t_samples = np.stack([_multivariate_t_rvs(comp.mu, comp.sigma, dof=comp.dof, n=N, random_state=rng) for comp in self.comps], axis=1)

    if rng is None:
        rng = np.random.RandomState()

    mask = rng.choice(range(len(self.weights)), size=N, p=self.weights)
    mask_array = np.eye(len(self.weights))[mask]
    mask_array = np.tile(np.expand_dims(mask_array, axis=-1), (1, 1, t_samples.shape[-1]))
    mixture_samples = np.reshape(np.sum(np.multiply(t_samples, mask_array), axis=1), (N, t_samples.shape[-1]))
    origin = np.logical_not(mask).astype(np.int)
    if trace is True:
        return mixture_samples, origin
    else:
        return mixture_samples

class _ImportanceSamplerTMixture(object):
    """ replaces the pypmc importance sampler with a more efficient implementation of importance
     sampling from a mixture of multivariate student-t distributions"""

    def __init__(self, target, proposal, indicator=None, prealloc=0,
                 save_target_values=False, rng=None):
        self.proposal = copy.deepcopy(proposal)
        self.rng = rng
        self.target = target
        self.target_values = _History(1, prealloc) if save_target_values else None
        self.weights = _History(1, prealloc)
        self.samples = _History(proposal.dim, prealloc)
        if rng is None: rng = np.random.RandomState(None)

    def clear(self):
        '''Clear history of samples and other internal variables to free memory.

        .. note::
            The init_proposal is untouched.

        '''
        self.samples.clear()
        self.weights.clear()
        if self.target_values is not None:
            self.target_values.clear()

    def run(self, N=1, trace_sort=False):
        '''Run the sampler, store the history of visited points into
        the member variable ``self.samples`` and the importance weights
        into ``self.weights``.

        .. seealso::
            :py:class:`pypmc.tools.History`

        :param N:

            Integer; the number of samples to be drawn.

        :param trace_sort:

            Bool; if True, return an array containing the responsible
            component of ``self.init_proposal`` for each sample generated
            during this run.

            .. note::

                This option only works for proposals of type
                :py:class:`pypmc.density.mixture.MixtureDensity`

            .. note::

                If True, the samples will be ordered by the components.

        '''
        if N == 0:
            return 0

        if trace_sort:
            this_samples, origin = self._get_samples(N, trace_sort=True)
            self._calculate_weights(this_samples, N)
            return origin
        else:
            this_samples = self._get_samples(N, trace_sort=False)
            self._calculate_weights(this_samples, N)

    def _calculate_weights(self, this_samples, N):
        """Calculate and save the weights of a run."""

        this_weights = self.weights.append(N)[:,0]

        if self.target_values is None:
            targets = self.target(this_samples)
            for i in range(N):
                tmp = targets[i] - self.proposal.evaluate(this_samples[i])
                this_weights[i] = _exp(tmp)
        else:
            this_target_values = self.target_values.append(N)
            targets = self.target(this_samples)
            for i in range(N):
                this_target_values[i] = targets[i]
                tmp = targets[i] - self.proposal.evaluate(this_samples[i])
                this_weights[i] = _exp(tmp)

    def _get_samples(self, N, trace_sort):
        """Save N samples from ``self.init_proposal`` to ``self.samples``
        This function does NOT calculate the weights.

        Return a reference to this run's samples in ``self.samples``.
        If ``trace_sort`` is True, additionally return an array
        indicating the responsible component. (MixtureDensity only)

        """
        # allocate an empty numpy array to store the run and append accept count
        # (importance sampling accepts all points)
        this_run = self.samples.append(N)

        sampler = _Student_t_Mixture_Density_Sampler(self.proposal)

        # store the proposed points (weights are still to be calculated)
        if trace_sort:
            this_run[:], origin = sampler.propose(N, self.rng, trace=True)
            return this_run, origin
        else:
            this_run[:] = sampler.propose(N, self.rng)
            return this_run


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