import unittest

import numpy as np

from cde.density_estimator import NeighborKernelDensityEstimation


class TestNeighborKernelDensityEstimation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def _generate_data(self, n_samples=300):
        X = np.random.normal(loc=0.0, scale=1e-2, size=(n_samples, 1))
        Y = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
        return X, Y

    def test_pdf_and_log_pdf_shapes(self):
        X, Y = self._generate_data()
        model = NeighborKernelDensityEstimation(
            name="nkde_test",
            ndim_x=1,
            ndim_y=1,
            epsilon=0.5,
            bandwidth=0.5,
            param_selection=None,
            weighted=False,
        )
        model.fit(X, Y)

        X_query = np.zeros((20, 1))
        Y_query = np.linspace(-1.5, 1.5, 20).reshape(-1, 1)

        pdf_vals = model.pdf(X_query, Y_query)
        log_pdf_vals = model.log_pdf(X_query, Y_query)

        self.assertEqual(pdf_vals.shape, log_pdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0.0))
        self.assertTrue(np.all(log_pdf_vals <= 0.0))

