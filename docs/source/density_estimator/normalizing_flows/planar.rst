Planar Flow
========================

The planar flow introduced in [REZENDE2015]_.
This flow was originally designed for variational inference and sampling.
Therefore it doesn't automatically fit our use-case of density estimation.
Since we especially need the inverse :math:`f^{-1}(x)` of the flow to be easily computable, we invert it's direction,
defining it as a mapping from the transformed distribution :math:`p_1(x)` to the base distribution :math:`p_0(x)`. Hence the flow is called
`InvertedPlanarFlow` in our implementation and the `forward` method is not implemented.

.. math::
    \mathbf{u},\mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}

    f^{-1}(x) = \mathbf{x} + \mathbf{u}\cdot\tanh(\mathbf{w}^T \mathbf{x} + b)

To make sure :math:`f(x)` exists, :math:`\mathbf{w}^T\mathbf{u} \geq-1` needs to hold.
Our implementation automatically constrains :math:`\mathbf{u}` before assignment using

.. math::
    \mathbf{û} = \mathbf{u} + (m (\mathbf{w}^T\mathbf{u}) - (\mathbf{w}^T \mathbf{u)})\dfrac{\mathbf{w}}{\|\mathbf{w}\|^2}

    m(\mathbf{x}) = -1 + softplus(\mathbf{x})

Dimension of parameter space: :math:`d + d+ 1`

Determinant of the Jacobian of :math:`f^{-1}(x)`:

.. math::
    \det(\mathbf{J}^{-1}) =
    |\mathbf{I} + \mathbf{u}^T \cdot \tanh'(\mathbf{w}^T \mathbf{x} + b) \cdot \mathbf{w}|

Hence the Inverse Log Det Jacobian for this flow is:

.. math::
    \log(\det(\mathbf{J}^{-1})) =
    \log(|\mathbf{I} + \mathbf{u}^T \cdot \tanh'(\mathbf{w}^T \mathbf{x} + b) \cdot \mathbf{w}|)


.. automodule:: cde.density_estimator.normalizing_flows

.. autoclass:: InvertedPlanarFlow
    :members:
    :inherited-members:

.. [REZENDE2015] Rezende, Mohamed (2015). Variational Inference with Normalizing Flows (http://arxiv.org/abs/1505.05770)
