Least-Squares Density Ratio Estimation
==================================================
Implementation of Least-Squares Density Ratio Estimation
(LS-CDE) method introduced in [SUG2010]_ with some minor extra features.



This approach estimates the conditional density of multi-dimensional
inputs/outputs by expressing the conditional density in terms of the ratio of
unconditional densities r(x,y):

.. math:: p(y|x) = \frac{p(x,y)}{p(x)} = r(x,y)

Instead of going through density estimation, this
work proposes to compute the density ratio function r(x,y) directly from
samples (x/y pairs of input/output). The density ratio function is modelled by
the following linear model:

.. math:: \widehat{r_{\alpha}}(x,y) := \alpha^T \phi(x,y)

while :math:`\alpha=(\alpha_1, \alpha_2,...,\alpha_b)^T`
being the parameters learned from samples and :math:`\phi(x,y) = (\phi_{1}(x,
y),\phi_{2}(x,y),...,\phi_{b}(x,y))^T` being basis functions such that
:math:`\phi(x,y) \geq 0_{b}` for all :math:`(x,y)\in D_{X} \times D_{Y}`.
:math:`0_{b}` denotes the b-dimensional vector with all zeros and
:math:`D_{X}`, :math:`D_{Y}` the input and output domains.


More precisely, the parameters :math:`\alpha` can be computed analytically by
minimizing a squared error :math:`\int\int\widehat{r_{\alpha}}(x,y) - r(x,y))
^2 p(x)dxdy`. However, to minimize this error, expectations over unknown
densities have to be computed which are approximated by the provided sample
averages. After having obtained the density-ratio function, the solution for
test time is denoted as:

.. math:: \widehat{p}(y|x=\tilde{x}) = \frac{\widehat{\alpha}^T\phi(\tilde{x},y)}{\int\widehat{\alpha}^T\phi(\tilde{x},y)dy}

while :math:`\tilde{x}` being the test input point. For the basis functions,
this work proposes to use a Gaussian kernel with width :math:`\sigma`
(bandwidth parameter) for both x and y. It further suggests to randomly
choose (x,y) center points, though in our implementation we offer various kernel
selection methods (all, random, distance, k_means, agglomerative). The work
also introduces a regularization parameter :math:`\lambda>0` for
stabilization purposes in the optimization. The model structure allows for
model selection by cross-validation.



.. automodule:: density_estimator

.. autoclass:: LSConditionalDensityEstimation
    :members:
    :inherited-members:


.. [SUG2010] http://proceedings.mlr.press/v9/sugiyama10a.html