[![Build Status](https://travis-ci.org/freelunchtheorem/Conditional_Density_Estimation.svg?branch=master)](https://travis-ci.org/freelunchtheorem/Conditional_Density_Estimation) [![Downloads](https://pepy.tech/badge/cde)](https://pepy.tech/project/cde)

# Conditional Density Estimation (CDE)

## Description
Implementations of various methods for conditional density estimation

* **Parametric neural network based methods**
    * Mixture Density Networks (MDN)
    * Kernel Density Estimation (KMN)
* **Nonparametric methods**
    * Conditional Kernel Density Estimation (CKDE)
    * Neighborhood Kernel Density Estimation (NKDE)
* **Semiparametric methods**
    * Least Squares Conditional Density Estimation (LSKDE)
    
Beyond estimating conditional probability densities, the package features extensive functionality for computing:
* **Centered moments:** mean, covariance, skewness and kurtosis
* **Statistical divergences:** KL-divergence, JS-divergence, Hellinger distance
* **Percentiles and expected shortfall**

## Installation
To use the library, either clone the GitHub repository and run 
```bash
pip install .
``` 
in the package folder or directly use the python package index:
```bash
pip install cde
```
Note that the package only supports tensorflow versions between 1.4 and 1.7.
## Documentation
See the documentation [here](https://freelunchtheorem.github.io/Conditional_Density_Estimation). A technical report & paper will soon be released and linked here.

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
If you use CDE in your research, you can cite it as follows:

```
@misc{cde2019,
    author = {Jonas Rothfuss, Fabio Ferreira},
    title = {Conditional Density Estimation},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ferreira-fabio/Conditional_Density_Estimation}},
}
```
