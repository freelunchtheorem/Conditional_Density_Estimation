Normalizing Flow Estimator
==================================================

The Normalizing Flow Estimator (NFE) combines a conventional neural network (in our implementation specified as :math:`estimator`) with a multi-stage Normalizing Flow [REZENDE2015]_ for modeling conditional probability distributions :math:`p(y|x)`.
Given a network and a flow, the distribution :math:`y` can be specified by having the network output the parameters of the flow given an input :math:`x` [TRIPPE2018]_.
If the normalizing flow is expressive enough, arbitrary conditional distributions can be approximated.

The flows work by transforming a base distribution (in our case a normal distribution) into successively more complex distributions
by applying bijectors.

Example of a normal distribution being transformed by two planar flows:

.. image:: normalizing_flows/planar_flow.png

Using the change of variable formula, the resulting probability distribution :math:`p_1` for a single flow :math:`f` applied to the base distribution :math:`p_0` becomes:

.. math::
    p_0(\mathbf{z_0}) = \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})(\mathbf{z_0})

    \mathbf{z_1} = f(\mathbf{z_0})

    p_1(\mathbf{z_1}) = p_0(f^{-1}(\mathbf{z_1})) \cdot |\det \dfrac{d f^{-1}(\mathbf{z_1})}{d \mathbf{z_1}}|

Using normalizing flows for density estimation requires that the inverse and the Jacobian determinant of the flow can be calculated quickly.

Given input :math:`x`, the neural network outputs the parameters :math:`\theta` of the flows.
The weights and biases :math:`w` of the neural network are learned by minimizing the negative logarithm of the likelihood (maximum likelihood) over :math:`N`
data points for a normalizing flow consisting of :math:`K` flows.

.. math::
    E(w) = - \sum_{n=1}^N \bigg\{\log p_0(\mathbf{z_{0,n}}) + \sum_{k=1}^{K} \log|\det\dfrac{d f_k^{-1}(\mathbf{z_{k,n}}, \theta_k(\mathbf{w}, \mathbf{x_n}))}{d \mathbf{z_{k,n}}}|\bigg\}

    \mathbf{z_{0,n}} = f_1^{-1}(f_2^{-1}(\dots f_K^{-1}(\mathbf{z_{K,n}}))), \mathbf{z_{K,n}} = \mathbf{y_n}

Available flows:

.. toctree::
    :maxdepth: 2
    :glob:

    ./normalizing_flows/*

.. automodule:: cde.density_estimator

.. autoclass:: NormalizingFlowEstimator
    :members:
    :inherited-members:

.. [REZENDE2015] Rezende, Mohamed (2015). Variational Inference with Normalizing Flows (http://arxiv.org/abs/1505.05770)
.. [TRIPPE2018] Trippe, Turner (2018). Conditional Density Estimation with Bayesian Normalising Flows (http://arxiv.org/abs/1802.04908)


