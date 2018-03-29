Neighborhood Kernel Density Estimation
==================================================

For estimating the conditional density :math:`p(y|x)`, :math:`\epsilon`-neighbor kernel density estimation (:math:`\epsilon`-KDE)
employs standard kernel density estimation in a local :math:`\epsilon`-neighborhood around a query point :math:`(x,y)`.

:math:`\epsilon`-KDE is a lazy learner, meaning that it simply stores the training points :math:`\{(x_i,y_i)\}_{i=1}^n`
and puts a kernel function in each of the points. In order to compute :math:`p(y|x)`, the estimator only considers a local
subset of the training samples :math:`\{(x_i, y_i)\}_{i \in \mathcal{I}_{x, \epsilon}}`, where :math:`\mathcal{I}_{x, \epsilon}` is the set
of sample indices such that :math:`||x_i - x|| \leq \epsilon`.

In case of Gaussian Kernels, the estimated density can be expressed as

.. math:: p(y|x) = \sum_{j \in \mathcal{I}_{x, \epsilon}} w_j ~ N(y~| y_j, \sigma^2 I)

where :math:`w_j` is the weighting of te j-th kernel and :math:`N(y~|\mu,\Sigma)` the probability function of a multivariate Gaussian.
This implementation currently supports two types of weighting:

- equal weights: :math:`w_j = \frac{1}{|\mathcal{I}_{x, \epsilon}|}`
- weights :math:`w_j` proportional to :math:`||x_j - x||_2`, the euclidean distance w.r.t. to x


.. automodule:: cde.density_estimator

.. autoclass:: NeighborKernelDensityEstimation
    :members:
    :inherited-members: