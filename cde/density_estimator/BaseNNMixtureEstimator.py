from abc import ABC, abstractmethod
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
import numpy as np

from cde.density_estimator.BaseNNEstimator import BaseNNEstimator


class BaseNNMixtureEstimator(BaseNNEstimator, ABC):
    """Mixin with helper methods for mixture-based estimators."""

    weight_decay = 0.0
    entropy_reg_coef = 0.0

    @abstractmethod
    def _get_mixture_components(self, X):
        """Return tuple (weights, locs, scales) describing the mixture for every X."""
        raise NotImplementedError()

    def mean_(self, x_cond, n_samples=None):
        assert hasattr(self, "_get_mixture_components")
        assert self.fitted, "model must be fitted"
        x_cond = self._handle_input_dimensionality(x_cond)
        means = np.zeros((x_cond.shape[0], self.ndim_y))
        weights, locs, _ = self._get_mixture_components(x_cond)
        for i in range(x_cond.shape[0]):
            means[i, :] = weights[i].dot(locs[i])
        return means

    def std_(self, x_cond, n_samples=2 * 10 ** 6):
        covs = self.covariance(x_cond, n_samples=n_samples)
        return np.sqrt(np.diagonal(covs, axis1=1, axis2=2))

    def covariance(self, x_cond, n_samples=None):
        assert self.fitted, "model must be fitted"
        x_cond = self._handle_input_dimensionality(x_cond)
        covs = np.zeros((x_cond.shape[0], self.ndim_y, self.ndim_y))
        glob_mean = self.mean_(x_cond)
        weights, locs, scales = self._get_mixture_components(x_cond)
        for i in range(x_cond.shape[0]):
            c1 = np.diag(weights[i].dot(scales[i] ** 2))
            c2 = np.zeros(c1.shape)
            for j in range(weights.shape[1]):
                a = locs[i][j] - glob_mean[i]
                d = weights[i][j] * np.outer(a, a)
                c2 += d
            covs[i] = c1 + c2
        return covs

    def mean_std(self, x_cond, n_samples=None):
        mean = self.mean_(x_cond, n_samples=n_samples)
        std = self.std_(x_cond, n_samples=n_samples)
        return mean, std

    def sample(self, X):
        assert self.fitted, "model must be fitted"
        assert self.can_sample
        X = self._handle_input_dimensionality(X)
        if np.all(np.all(X == X[0, :], axis=1)):
            return self._sample_rows_same(X)
        return self._sample_rows_individually(X)

    def conditional_value_at_risk(self, x_cond, alpha=0.01, n_samples=10**7):
        assert self.fitted, "model must be fitted"
        assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
        x_cond = self._handle_input_dimensionality(x_cond)
        assert x_cond.ndim == 2
        VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
        return self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)

    def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 7):
        assert self.fitted, "model must be fitted"
        assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
        assert x_cond.ndim == 2
        VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
        CVaRs = self._conditional_value_at_risk_mixture(VaRs, x_cond, alpha=alpha)
        assert VaRs.shape == CVaRs.shape == (len(x_cond),)
        return VaRs, CVaRs

    def _conditional_value_at_risk_mixture(self, VaRs, x_cond, alpha=0.01):
        weights, locs, scales = self._get_mixture_components(x_cond)
        locs = locs.reshape(locs.shape[:2])
        scales = scales.reshape(scales.shape[:2])
        CVaRs = np.zeros(x_cond.shape[0])
        c = (VaRs[:, None] - locs) / scales
        for i in range(x_cond.shape[0]):
            cdf = norm.cdf(c[i])
            pdf = norm.pdf(c[i])
            cdf = np.ma.masked_where(cdf < 10 ** -64, cdf)
            pdf = np.ma.masked_where(pdf < 10 ** -64, pdf)
            CVaRs[i] = np.sum((weights[i] * cdf / alpha) * (locs[i] - scales[i] * (pdf / cdf)))
        return CVaRs

    def _sample_rows_same(self, X):
        weights, locs, scales = self._get_mixture_components(np.expand_dims(X[0], axis=0))
        weights = weights.astype(np.float64)
        weights = weights / np.sum(weights)
        gmm = GaussianMixture(n_components=self.n_centers, covariance_type="diag", max_iter=5, tol=1e-1)
        gmm.fit(np.random.normal(size=(100, self.ndim_y)))
        gmm.converged_ = True
        gmm.weights_ = weights[0]
        gmm.means_ = locs[0]
        gmm.covariances_ = scales[0]
        y_sample, _ = gmm.sample(X.shape[0])
        return X, y_sample

    def _sample_rows_individually(self, X):
        weights, locs, scales = self._get_mixture_components(X)
        Y = np.zeros(shape=(X.shape[0], self.ndim_y))
        for i in range(X.shape[0]):
            idx = np.random.choice(range(locs.shape[1]), p=weights[i, :])
            Y[i, :] = np.random.normal(loc=locs[i, idx, :], scale=scales[i, idx, :])
        return X, Y

    def cdf(self, X, Y):
        assert self.fitted, "model must be fitted"
        assert hasattr(self, "_get_mixture_components"), "cdf computation requires _get_mixture_components method"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        weights, locs, scales = self._get_mixture_components(X)
        P = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(self.n_centers):
                P[i] += weights[i, j] * multivariate_normal.cdf(
                    Y[i], mean=locs[i, j, :], cov=np.diag(scales[i, j, :])
                )
        return P

    def reset_fit(self):
        super().reset_fit()

