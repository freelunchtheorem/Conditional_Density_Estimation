import unittest

import numpy as np

from cde.density_estimator import ConditionalKernelDensityEstimation


class TestConditionalKernelDensityEstimation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def _generate_data(self, n_samples=400):
        X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
        Y = X + np.random.normal(scale=0.3, size=(n_samples, 1))
        return X, Y

    def test_pdf_cdf_after_fit(self):
        X, Y = self._generate_data()
        model = ConditionalKernelDensityEstimation(
            name="ckde_test",
            ndim_x=1,
            ndim_y=1,
            bandwidth='normal_reference',
        )
        model.fit(X, Y)

        X_eval = np.zeros((25, 1))
        Y_eval = np.linspace(-2.0, 2.0, 25).reshape(-1, 1)

        pdf_vals = model.pdf(X_eval, Y_eval)
        cdf_vals = model.cdf(X_eval, Y_eval)

        self.assertEqual(pdf_vals.shape, cdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0.0))
        self.assertTrue(np.all((cdf_vals >= 0.0) & (cdf_vals <= 1.0)))


