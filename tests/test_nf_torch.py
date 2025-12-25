import unittest

import numpy as np
import torch

from cde.density_estimator import NormalizingFlowEstimator


class TestNormalizingFlowTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def _generate_data(self, n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
        Y = 3.0 * X.squeeze() + np.random.normal(scale=0.2, size=X.shape[0])
        return X, Y.reshape(-1, 1)

    def test_training_log_pdf_pdf_cdf_shapes(self):
        X, Y = self._generate_data()
        model = NormalizingFlowEstimator(
            name="nf_torch_test",
            ndim_x=1,
            ndim_y=1,
            flows_type=("affine", "radial"),
            hidden_sizes=(16,),
            hidden_nonlinearity="tanh",
            n_training_epochs=120,
            batch_size=64,
            learning_rate=1e-2,
            data_normalization=True,
            dropout=0.0,
        )
        model.fit(X, Y, verbose=False)

        X_eval = np.linspace(-1.5, 2.0, 40).reshape(-1, 1)
        Y_eval = 3.0 * X_eval.squeeze() + np.linspace(-0.5, 0.5, X_eval.shape[0])

        pdf_vals = model.pdf(X_eval, Y_eval)
        log_pdf_vals = model.log_pdf(X_eval, Y_eval)
        cdf_vals = model.cdf(X_eval, Y_eval)

        self.assertEqual(pdf_vals.shape, log_pdf_vals.shape)
        self.assertEqual(pdf_vals.shape, cdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0))
        self.assertTrue(np.all((cdf_vals >= 0) & (cdf_vals <= 1)))

