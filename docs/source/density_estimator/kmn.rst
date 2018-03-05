Kernel Mixture Network
==========================================================

Implementation of Kernel Mixture Network introduced in [AMB2017]_ with some extra features.

The approach combines unconditional kernel density estimation with a (softmax) neural network,
obtaining a conditional kernel density estimator.
Comparable to unconditional kernel density estimation, kernels are placed in each of the training samples or a subset of
the samples. A neural network predicts the weights of the kernels based on the x (value to condition on) which
it receives as an input. Overall the the conditional probability density function is modeled as follows:

.. math::
    f(y|x) = \frac{1}{\sum_{p,j} w_{pj}(x; W)} \sum_{p,j} w_{pj}(x; W) \mathcal{K}_j(y,y^{(p)})

This implementation uses Gaussian Kernels:

.. math::
    \mathcal{K}(y,y';\sigma)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{\left\Vert y-y'\right\Vert^2}{2\sigma^2}}

In addition approach described in the paper, the implementation has the following extensions:

- **Trainable scales/bandwiths:** The scales of the Gaussian kernels can be either be fixed or jointly trained with
  the neural network weights. This property is controlled by the boolean train_scales in the constructor.

- Center Sampling Methods:
    - **all:** use all data points in the train set as kernel centers
    - **random:** randomly selects k points as kernel centers
    - **k_means:** uses k-means clustering to determine k kernel centers
    - **agglomorative:** uses agglomorative clustering to determine k kernel centers


.. automodule:: density_estimator

.. autoclass:: KernelMixtureNetwork
    :members:

The core of the Kernel Mixture Network implementation is originally written by [VEG2017]_.
In addition to the original implementation of Jan van der Vegt and Alexander Backus we added support for
mulivariate distributions p(y|x) as well as automated hyperparameter search via cross-validation.

.. [AMB2017] https://arxiv.org/abs/1705.07111
.. [VEG2017] https://github.com/janvdvegt/KernelMixtureNetwork