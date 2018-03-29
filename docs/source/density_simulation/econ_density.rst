ECON Toy Density Model
==========================================================

A simple, economically inspired distribution with the data generating process

.. math:: x = |\epsilon_y|, ~~~ \epsilon_x \sim N(0,1)
.. math:: y = x^2 + \epsilon_y, ~~~ \epsilon_y \sim N(0,\sigma_y)

If heteroscedastic = True, :math:`\sigma_y` is a linear function of x:

.. math:: \sigma_y = 1 + x

.. automodule:: cde.density_simulation

.. autoclass:: EconDensity
    :members:
    :inherited-members: