Affine Flow
========================

A scale-and-shift bijector

.. math::
    a,b \in \mathbb{R}^d

    f(x) = \exp(a^T) \cdot x + b

Dimension of parameter space: :math:`d+d`

Determinant of the Jabobian:

.. math::
    \det(\mathbf{J}) =
    \prod_{j=1}^{d}\exp(\mathbf{a}_j)
    = \exp(\sum_{j=1}^{d}\mathbf{a}_j)

Hence the Inverse Log Det Jacobian is:

.. math::
    \log(\det(\mathbf{J}^{-1}))
    = -(\sum_{j=1}^{d}\mathbf{a}_j)


.. automodule:: cde.density_estimator.normalizing_flows

.. autoclass:: AffineFlow
    :members:
    :inherited-members: