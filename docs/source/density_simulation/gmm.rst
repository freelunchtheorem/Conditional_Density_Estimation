Gaussian Mixture
==========================================================

Fit a and sample from a uni- bi- or multivariate Gaussian mixture model with diagonal covariance matrices. For the multivariate case the distribution is given by

.. math:: G(X | \mu, \Sigma) = \frac{1}{\sqrt{2\pi\left|\Sigma\right|}} \exp^{(-\frac{1}{2} (X-\mu)^T\Sigma^{-1}(X-\mu))} 

The mixture model is then composed of a linear combination of an arbitrary number of components :math:`K`:

.. math:: p(X) = \sum_{k=1}^K \pi_k G(X|\mu_k, \Sigma_k).

Where :math:`\pi_k` is the mixing coefficient for the :math:`k`-th distribution. :math:`\mu`, :math:`\Sigma` and :math:`\pi` are estimated by Maximum-Likelihood for each :math:`k`. It is possible to specify the number of kernels to define the modality of the distribution and also dimensionality for both :math:`x` and :math:`y`. The component means are initialized randomly according to given standard deviation. Also the weights are initialized randomly.

.. automodule:: cde.density_simulation

.. autoclass:: GaussianMixture
    :members:
    :inherited-members: