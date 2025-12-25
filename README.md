[![Build Status](https://travis-ci.org/freelunchtheorem/Conditional_Density_Estimation.svg?branch=master)](https://travis-ci.org/freelunchtheorem/Conditional_Density_Estimation) [![Downloads](https://pepy.tech/badge/cde)](https://pepy.tech/project/cde)

# Conditional Density Estimation (CDE)

**Update:** Conditional Density Estimation now runs with PyTorch on the `pytorch-migration` (soon `main`) branch; the legacy TensorFlow implementation lives in the `tensorflow` branch. All core estimators, runners, and examples are tested with the latest PyTorch release.

## Description
Implementations of various methods for conditional density estimation

* **Parametric neural network based methods**
    * Mixture Density Network (MDN)
    * Kernel Mixture Network (KMN)
    * Normalizing Flows (NF)
* **Nonparametric methods**
    * Conditional Kernel Density Estimation (CKDE)
    * Neighborhood Kernel Density Estimation (NKDE)
* **Semiparametric methods**
    * Least Squares Conditional Density Estimation (LSKDE)
    
Beyond estimating conditional probability densities, the package features extensive functionality for computing:
* **Centered moments:** mean, covariance, skewness and kurtosis
* **Statistical divergences:** KL-divergence, JS-divergence, Hellinger distance
* **Percentiles and expected shortfall**

For the parametric models (MDN, KMN, NF), we recommend the usage of noise regularization which is supported by our implementation. For details, we refer to the paper [Noise Regularization for Conditional Density Estimation](https://arxiv.org/abs/1907.08982).

## Installation

Clone the repository and run the provided script to create the `cde-pytorch` Conda environment (Python 3.11/3.10 with CPU PyTorch plus the pinned NumPy/SciPy versions that are tested with CDE):
```bash
bash scripts/setup_pytorch_env.sh
```
After you activate the environment, install the local package in editable mode:
```bash
pip install --break-system-packages -e .
```
If you already have a PyTorch environment, you can install the package with `pip install cde`; the runtime expects the usual scientific stack (`numpy`, `scipy`, `pandas`, `matplotlib`) and `ml_logger`.
## Documentation and paper
See the documentation [here](https://freelunchtheorem.github.io/Conditional_Density_Estimation). A paper on best practices and benchmarks on conditional density estimation with neural networks that makes extensive use of this library can be found [here](https://arxiv.org/abs/1903.00954).

## Usage
The following code snipped holds an easy example that demonstrates how to use the cde package.
```python
from cde.density_simulation import SkewNormal
from cde.density_estimator import KernelMixtureNetwork
import numpy as np

""" simulate some data """
density_simulator = SkewNormal(random_seed=22)
X, Y = density_simulator.simulate(n_samples=3000)

""" fit density model """
model = KernelMixtureNetwork("KDE_demo", ndim_x=1, ndim_y=1, n_centers=50,
                             x_noise_std=0.2, y_noise_std=0.1, random_seed=22)
model.fit(X, Y)

""" query the conditional pdf and cdf """
x_cond = np.zeros((1, 1))
y_query = np.ones((1, 1)) * 0.1
prob = model.pdf(x_cond, y_query)
cum_prob = model.cdf(x_cond, y_query)

""" compute conditional moments & VaR  """
mean = model.mean_(x_cond)[0][0]
std = model.std_(x_cond)[0][0]
skewness = model.skewness(x_cond)[0]
```
## Citing
If you use our CDE implementation in your research, you can cite it as follows:

```
@article{rothfuss2019conditional,
  title={Conditional Density Estimation with Neural Networks: Best Practices and Benchmarks},
  author={Rothfuss, Jonas and Ferreira, Fabio and Walther, Simon and Ulrich, Maxim},
  journal={arXiv:1903.00954},
  year={2019}
}

```
If you use noise regularization for regularizing the MDN, KMN or NF conditional density model, please cite
```
@article{rothfuss2019noisereg,
    title={Noise Regularization for Conditional Density Estimation},
    author={Jonas Rothfuss and Fabio Ferreira and Simon Boehm and Simon Walther 
            and Maxim Ulrich and Tamim Asfour and Andreas Krause},
    year={2019},
    journal={arXiv:1907.08982},
}
```

## Todo
- track configuration for the new PyTorch branch and keep the legacy TensorFlow branch discoverable
