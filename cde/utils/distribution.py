import numpy as np
from scipy.special._ufuncs import gamma

""" Multivariate Student-t pdf and rvs """


def _multidim_t_pdf(x, mu, sigma, dof):
    '''
    Multidimensional t-student density:

    Args:
        x: points where to calculate the pdf - array of shape (batch_size, ndim_x)
        mu: mean - array of shape (ndim_x, )
        sigma: scale -  array of shape (ndim_x, )
        dof = degrees of freedom
        d: dimension

    Returns:
        p: probability density p(x) - array of shape (batch_size)
    '''
    d = mu.shape[0]
    num = gamma((d + dof) / 2.0)
    denom = gamma(dof / 2.0) * (dof * np.pi) ** (d / 2.0) * np.prod(sigma) ** 0.5 * \
            (1 + (1. / dof) * np.sum((x - mu)**2/sigma, axis=-1))**((d + dof) / 2.0)
    p = num / denom
    assert p.ndim == 1
    return p


def _multidim_t_rvs(mu, sigma, dof, N=1, random_state=None):
    ''' generates random variables of multidmensional (diagonal covariance matrix)
        t distribution

    Args:
        mu = mean - array of shape (ndim_x, )ble
        sigma: scale -  array of shape (ndim_x, )
        dof: (numeric) degrees of freedom
        N: (int) number of observations, return random array will be (n, ndim_x))
        random_state: (np.random.RandomState) random number generator object

    Returns:
        rvs: ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
    '''

    return _multivariate_t_rvs(mu, np.diag(sigma), dof, N, random_state=random_state)


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