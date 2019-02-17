[![Build Status](https://travis-ci.org/ferreira-rothfuss/Conditional_Density_Estimation.svg?branch=master)](https://travis-ci.org/ferreira-rothfuss/Conditional_Density_Estimation)

# Conditional Density Estimation (CDE)

## Description
Implementations of various methods for conditional density estimation

* **Parametric neural network based methods**
    * Mixture Density Networks (MDN)
    * Kernel Density Estimation (KMN)
* **Nonparametric methods**
    * Conditional Kernel Density Estimation (CKDE)
    * $\epsilon$-Neighborhood Kernel Density Estimation (NKDE)
* **Semiparametric methods**
    * Least Squares Conditional Density Estimation (LSKDE)
    
Beyond estimating conditional probability densities, the package features extensive functionality for computing:
* **Centered moments:** mean, covariance, skewness and kurtosis
* **Statistical divergences:** KL-divergence, JS-divergence, Hellinger distance
* **Percentiles and expected shortfall**

## Installation
To use the library, either clone the GitHub repository and import it into your projects or install the pip package:
```
pip install cde
```
## Documentation
See the documentation [here](https://jonasrothfuss.github.io/Conditional_Density_Estimation).


## Citing
If you use CDE in your research, you can cite it as follows:

```
@misc{cde2019,
    author = {Jonas Rothfuss, Fabio Ferreira},
    title = {Conditional Density Estimation},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/jonasrothfuss/Conditional_Density_Estimation}},
}
```
