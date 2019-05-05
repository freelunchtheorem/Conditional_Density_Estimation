Skew Normal
==========================================================

This model is a univariate skewed normal distribution motivated by Azzalini et al. (2003), "Graphical models for skew-normal variates" and is defined by

.. math:: x \in N(0, 0.5)
.. math:: \mu = a*x+b
.. math:: \sigma = c*x^2+d
.. math:: \alpha = skew_{high} + sigmoid(x) * (skew_{high}-skew_{low})
.. math:: y = 2*PDF(x;\mu,\sigma^2)*CDF(x\alpha)

While *PDF* and *CDF* referring to the probability density and cumulative distribution function of a normally distributed random variable with :math:`skew_{high} = -4` and :math:`skew_{low}=-0.1`.

.. automodule:: cde.density_simulation

.. autoclass:: SkewNormal
    :members:
    :inherited-members: