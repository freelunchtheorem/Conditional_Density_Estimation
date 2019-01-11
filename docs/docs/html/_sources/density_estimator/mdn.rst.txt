Mixture Density Network
==================================================

The Mixture Density Network (MDN) [BISHOP1994]_ combines a conventional neural network (in our implementation specified as :math:`estimator`) with a mixture
density model for modeling conditional probability distributions :math:`p(t|x)`. Given a sufficiently flexible network and considering a parametric mixture
model, the parameters of the distribution :math:`t` can be determined by the outputs of the neural network provided the input to the network is :math:`x` (in
our implementation specified as X_ph (placeholder) and X). This approach therefore constitutes as a framework capable of approximating
arbitrary conditional distributions.

The following example develops a model for Gaussian components with isotropic component covariances, while :math:`K`
being the number of components of a single mixture (our model currently currently allows to choose an arbitrary number of (global) mixture components, see
parameter L below) and :math:`\pi(x)` denoting the mixing coefficients:

.. math:: p(t|x) = \sum_{k=1}^K \pi_{k}(x) \mathcal{N}(t|\mu_{k}(x), \sigma_{k}^2(x))

It is both feasible to replace the components by components of other distributions and extending the MDN to arbitrary covariance matrices.
Although the later is generally much more difficult, it has been shown by [TANSEY2016]_ that, for example one can have the MDN output the lower
triangular entries in the Cholesky decomposition.

Using :math:`x` as input, the mixing coefficients :math:`\pi_{k}(x)`, the means :math:`\mu_{k}(x)`, and the variances :math:`\sigma_{k}^2(x)` can be
governed by the outputs of neural network. Assuming the mixture model has L mixture components (in our implementation specified as n_centers), the total number
of network outputs is given by :math:`(K+2)L`.

The mixing coefficients are computed as a set of :math:`L` softmax outputs, where :math:`a_k^{\pi}` determine the mixing coefficients emitted by the network:

.. math:: \pi_k(x) = \frac{exp(a_k^{\pi})}{\sum_{l=1}^K exp(a_k^{\pi})}


ensuring the constraint that :math:`\pi_k(x)` over :math:`K` must sum to 1. Similarly, the variances must me larger or equal to zero. Due to isotropy we have
:math:`L` kernel widths :math:`\sigma_k(x)` which are determined by the network output :math:`a_k^{\sigma}` and can be represented as exponentials:

.. math::  \sigma_k(x) = exp(a_k^{\sigma})

For the :math:`K \times L` means we directly use the network outputs: :math:`\mu_k(x) = a_{kj}^{\sigma}.


The weights and biases :math:`w` of the neural network are learned by minimizing the negative logarithm of the likelihood (maximum likelihood) over :math:`N
data points:

.. math:: E(w) = - \sum_{n=1}^N \ln \bigg\{\sum_{k=1}^k \pi_k(x_n, w) \mathcal{N} (t_n|\mu_k(x_n, w), \sigma_k^2(x_n,w)) \bigg\}

This can be executed via the standard backpropagation algorithm, given that suitable expressions for the derivations can be obtained.


.. automodule:: cde.density_estimator

.. autoclass:: MixtureDensityNetwork
    :members:
    :inherited-members:


.. [BISHOP1994] Bishop (1994). Mixture Density Networks, Technical Report, Aston University (http://publications.aston.ac.uk/373/)
.. [TANSEY2016] Tansey et al. (2016). Better Conditional Density Estimation for Neural Networks (https://arxiv.org/abs/1606.02321)