ARMA-Jump Time Series Model
==========================================================

AR(1) model with jump component

Data generation process:

.. math:: x_t = \left[ c(1-\alpha) + \alpha x_{t-1} \right] + (1-z_t) \sigma \epsilon_t + z_t \left[ -3c + 2\sigma \epsilon_t \right]

where :math:`\epsilon_t \sim N(0,1)` denotes a Gaussian shock and :math:`z_t \sim B(1,p)` a Bernoulli distributed jump indicator with :math:`p`
being the probability for a negative jump.

.. automodule:: density_simulation

.. autoclass:: ArmaJump
    :members: