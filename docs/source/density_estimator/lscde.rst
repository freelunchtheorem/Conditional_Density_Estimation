Least-Squares Density Ratio Estimation
==================================================
Implementation of Least-Squares Density Ratio Estimation
(LS-CDE) method introduced in [SUG2010]_ with some extra features.


This approach estimates the conditional density of multi-dimensional
inputs/outputs by expressing the conditional density in terms of the ratio of
unconditional densities r(x,y):

.. math:: p(y|x) = \frac{p(x,y)}{p(x)} = r(x,y)

Instead of estimating both unconditional densities separately, the density ratio function r(x,y) is directly estimated from
samples. The density ratio function is modelled by the following linear model:

.. math:: \widehat{r_{\alpha}}(x,y) := \alpha^T \phi(x,y)

where :math:`\alpha=(\alpha_1, \alpha_2,...,\alpha_b)^T`
are the parameters learned from samples and :math:`\phi(x,y) = (\phi_{1}(x,
y),\phi_{2}(x,y),...,\phi_{b}(x,y))^T` are kernel functions such that
:math:`\phi_{l}(x,y) \geq 0` for all :math:`(x,y)\in D_{X} \times D_{Y}` and :math:`l = 1, ..., b`.


The parameters :math:`\alpha` are learned by minimizing the
a integrated squared error.

.. math:: J(\alpha) = \int\int ( \widehat{r_{\alpha}}(x,y) - r(x,y))^2 p(x)dxdy.

After having obtained :math:`\widehat{\alpha} = argmin_{\alpha} ~ J(\alpha)` through training, the conditional density can be computed as follows:

.. math:: \widehat{p}(y|x=\tilde{x}) = \frac{\widehat{\alpha}^T\phi(\tilde{x},y)}{\int\widehat{\alpha}^T\phi(\tilde{x},y)dy}
   :label: quotient

[SUG2010]_ propose to use a Gaussian kernel with width :math:`\sigma`
(bandwidth parameter), which is also the choice for this implementation:

.. math:: \phi_{l}(x,y) = exp \left( \frac{||x-u_{l}||^2}{2 \sigma^2} \right)  exp \left( \frac{||y-v_{l}||^2}{2 \sigma^2} \right)

where :math:`\{(u_{l},v_{l})\}_{l=1}^b` are center points that are chosen from the training data set.
By using Gaussian kernels the optimization problem :math:`argmin_{\alpha} ~ J(\alpha)` can be solved analytically.
Also, the denominator in :eq:`quotient` is traceable and can be computed analytically.
The fact that training does not require numerical optimization and the solution can be computed fully analytically is the key advantage of LS-CDE.


While [SUG2010]_ propose to select center points for the kernel functions randomly from the training set, our implementation offers further center sampling methods:

- **all:** use all data points in the train set as kernel centers
- **random:** randomly selects k points as kernel centers
- **k_means:** uses k-means clustering to determine k kernel centers
- **agglomorative:** uses agglomorative clustering to determine k kernel centers



.. automodule:: density_estimator

.. autoclass:: LSConditionalDensityEstimation
    :members:
    :inherited-members:

.. [SUG2010] Sugiyama et al. (2010). Conditional Density Estimation via Least-Squares Density Ratio Estimation, in PMLR 9:781-788 (http://proceedings.mlr.press/v9/sugiyama10a.html)