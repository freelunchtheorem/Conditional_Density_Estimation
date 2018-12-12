import copy
import numpy as np
import scipy.stats as stats
from math import exp as _exp
from pypmc.tools import History as _History

from pypmc.mix_adapt.pmc import student_t_pmc
from pypmc.density.mixture import create_t_mixture

""" Some Helper Functions for the Adaptive Monte Carlo Integration"""

def _multivariate_t_rvs(loc, cov, dof=np.inf, n=1, random_state=None):
    ''' generates random variables of multivariate t distribution
        Parameters

    Args:
        loc:  (array_like) mean of random variable, length determines dimension of random variable
        cov: (array_like) square array of covariance  matrix
        dof: (numeric) degrees of freedom
        n: (int) number of observations, return random array will be (n, len(m))
        random_state: (np.random.RandomState) random number generator object

    Returns:
        rvs: ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
    '''
    if random_state is None:
        random_state = np.random.RandomState(None)
    loc = np.asarray(loc)
    d = len(loc)
    if dof == np.inf:
        x = 1.
    else:
        x = random_state.chisquare(dof, n) / dof
    z = random_state.multivariate_normal(np.zeros(d), cov, (n,))
    return loc + z / np.sqrt(x)[:, None]

class _Student_t_Mixture_Density_Sampler():
  def __init__(self, student_t_mixture):
    assert len(student_t_mixture.components)
    self.weights = student_t_mixture.weights
    self.comps = student_t_mixture.components
    assert np.size(self.weights) == 2, "only supports mixture of two dists"

  def propose(self, N, rng=None, trace=False):
    t_samples = np.stack([_multivariate_t_rvs(comp.mu, comp.sigma, dof=comp.dof, n=N, random_state=rng) for comp in self.comps])

    mask = stats.bernoulli.rvs(self.weights[0], size=N, random_state=rng)
    mask_array = np.stack([mask, np.logical_not(mask)])
    mask_array = np.tile(np.expand_dims(mask_array, axis=-1), (1, 1, t_samples.shape[-1]))
    mixture_samples = np.reshape(np.sum(np.multiply(t_samples, mask_array), axis=0), (N, t_samples.shape[-1]))
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
            The proposal is untouched.

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
            component of ``self.proposal`` for each sample generated
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
        """Save N samples from ``self.proposal`` to ``self.samples``
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


""" The actual monte_carlo_integration function"""

def monte_carlo_integration(fun, log_prob, ndim, n_samples, adaptive=True, n_em_steps=4,
                            n_comps=2, random_state=None):
    """
    Estimates the integral $\int f(x) * p(x) dx = E_p[f(x)]$ with (adaptive) importance sampling.
    The proposal distribution is a mixture of student-t distributions and is adapted with PMC
    (https://link.springer.com/article/10.1007/s11222-008-9059-)
    Args:
        fun: (callable) function that receives a numpy array and returns a numpy array with one dimension
        log_prob: (callable) log_probability log p(x)
        ndim: (int) number of dimensions of x
        n_samples: (int) number of samples
        adaptive: (bool) whether to use an adaptive proposal distribution
                    If true, the population monte carlo (PMC) algorithm is used to optimize the proposal dist
        n_em_steps: (int) Number of expectation maximization steps for the PMC
    Returns:
        (float) estimate of the integral
    """
    log_target_adaptation = lambda x: log_prob(x).flatten() + np.log(np.abs(fun(x).flatten()))

    if random_state is None: random_state = np.random.RandomState(None)

    # create initial proposal dist (mixture of student_t distributions)
    dofs = random_state.uniform(low=2.5, high=20.0, size=n_comps) # sample dof of initial components
    weights = random_state.uniform(low=0.0, high=1.0, size=n_comps)
    weights = weights / np.mean(weights)
    means = [random_state.normal(loc=np.zeros(ndim), scale=0.01) for _ in range(n_comps)]
    covs = [np.eye(ndim) for _ in range(n_comps)]
    proposal_dist = create_t_mixture(means, covs, dofs, weights)

    sampler = _ImportanceSamplerTMixture(log_target_adaptation, proposal_dist, rng=random_state)

    # adapt proposal distribution
    if adaptive:
        generating_components = []
        for i in range(n_em_steps):
            generating_components.append(sampler.run(10 ** 4, trace_sort=True))
            samples, weights = sampler.samples[-1], sampler.weights[-1].flatten()
            print(proposal_dist.weights, [comp.mu for comp in proposal_dist.components], [comp.sigma for comp in proposal_dist.components])
            proposal_dist = student_t_pmc(samples, sampler.proposal,
                                    weights,
                                    latent=generating_components[-1],
                                    mincount=20, rb=True, copy=False)

    # monte carlo integration with importance sampling from proposal distribution
    log_target = lambda x: log_prob(x).flatten()
    sampler = _ImportanceSamplerTMixture(log_target, proposal_dist, rng=random_state)
    sampler.run(N=n_samples)
    samples, weights = sampler.samples[-1], sampler.weights[-1]
    result = np.mean(np.multiply(weights.flatten(), fun(samples).flatten()))
    return result