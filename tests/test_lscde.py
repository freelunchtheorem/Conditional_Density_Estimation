import unittest

import numpy as np

from cde.density_estimator import LSConditionalDensityEstimation


class TestLSConditionalDensityEstimation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def _generate_data(self, n_samples=500):
        X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
        Y = 2.5 * X.squeeze() + np.random.normal(scale=0.25, size=X.shape[0])
        return X, Y.reshape(-1, 1)

    def test_pdf_logpdf_after_fit(self):
        X, Y = self._generate_data()
        model = LSConditionalDensityEstimation(
            name="lscde_test",
            ndim_x=1,
            ndim_y=1,
            center_sampling_method="k_means",
            bandwidth=0.5,
            n_centers=50,
            regularization=0.1,
            keep_edges=False,
        )
        model.fit(X, Y)

        X_eval = np.zeros((25, 1))
        Y_eval = np.linspace(-2.0, 2.0, 25).reshape(-1, 1)

        pdf_vals = model.pdf(X_eval, Y_eval)
        log_pdf_vals = model.log_pdf(X_eval, Y_eval)

        self.assertEqual(pdf_vals.shape, log_pdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0.0))
        self.assertTrue(np.all(np.isfinite(log_pdf_vals)))

