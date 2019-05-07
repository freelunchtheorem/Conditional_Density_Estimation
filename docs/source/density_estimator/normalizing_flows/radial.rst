Radial Flow
========================

The radial flow introduced in [REZENDE2015]_ with the parametrization used in [TRIPPE2018]_.
The flow was originally designed for variational inference and sampling. Therefore it doesn't easily fit our use-case
of density estimation.
Since we especially need the inverse :math:`f^{-1}(x)` of the flow to be easily computable, we invert it's direction,
defining it as a mapping from the transformed distribution :math:`p_1(x)` to the base distribution :math:`p_0(x)`.
Hence the flow is called `InvertedRadialFlow` in our implementation and the `forward` method is not implemented.

.. math::
    \mathbf{\gamma} \in \mathbb{R}^d, \alpha, \beta \in \mathbb{R}

    f^{-1}(\mathbf{x}) = \mathbf{x} + \dfrac{\alpha\beta(\mathbf{x} - \mathbf{\gamma})}{\alpha + |\mathbf{x}-\mathbf{\gamma}|}

To ensure :math:`f(x)` exists we have to constrain the parameters of the flow:

- :math:`\alpha \geq 0` needs to hold. Therefore we apply a softplus transformation to :math:`\alpha`
- :math:`\beta \geq -1` needs to hold. We apply :math:`f(x) = \exp(x) - 1` to :math:`\beta` before assignment

Jacobian determinant:

.. math::
    \det(\mathbf{J}^{-1}) =
    \lbrack1 + \alpha\beta \cdot h(\alpha, r)\rbrack^{d -1} \lbrack1 + \alpha\beta \cdot h(\alpha, r) + \alpha\beta \cdot h'(\alpha, r)r\rbrack

    h(\alpha, r) = \dfrac{1}{\alpha + r}, r = |\mathbf{x} - \mathbf{\gamma}|

.. automodule:: cde.density_estimator.normalizing_flows

.. autoclass:: InvertedRadialFlow
    :members:
    :inherited-members:

.. [REZENDE2015] Rezende, Mohamed (2015). Variational Inference with Normalizing Flows (http://arxiv.org/abs/1505.05770)
.. [TRIPPE2018] Trippe, Turner (2018). Conditional Density Estimation with Bayesian Normalising Flows (http://arxiv.org/abs/1802.04908)
