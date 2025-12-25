import unittest

import numpy as np
import torch

from cde.density_estimator import MixtureDensityNetwork


class TestMixtureDensityNetworkTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def _generate_data(self, n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
        Y = 2.0 * X.squeeze() + np.random.normal(scale=0.15, size=X.shape[0])
        return X, Y.reshape(-1, 1)

    def test_training_pdf_logpdf_shapes(self):
        X, Y = self._generate_data()
        model = MixtureDensityNetwork(
            name="mdn_torch_test",
            ndim_x=1,
            ndim_y=1,
            n_centers=8,
            hidden_sizes=(32,),
            hidden_nonlinearity="tanh",
            n_training_epochs=200,
            batch_size=64,
            learning_rate=1e-2,
            data_normalization=True,
            dropout=0.0,
        )
        model.fit(X, Y, verbose=False)

        X_eval = np.linspace(-1.5, 2.0, 40).reshape(-1, 1)
        Y_eval = 2.0 * X_eval.squeeze() + np.linspace(-0.5, 0.5, X_eval.shape[0])

        pdf_vals = model.pdf(X_eval, Y_eval)
        log_pdf_vals = model.log_pdf(X_eval, Y_eval)
        cdf_vals = model.cdf(X_eval, Y_eval)

        self.assertEqual(pdf_vals.shape, log_pdf_vals.shape)
        self.assertEqual(pdf_vals.shape, cdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0))
        self.assertTrue(np.all((cdf_vals >= 0) & (cdf_vals <= 1)))

        sampled = model.sample(X_eval)
        self.assertEqual(sampled[0].shape, X_eval.shape)
        self.assertEqual(sampled[1].shape, X_eval.shape)

        mean_vals = model.mean_(X_eval)
        self.assertEqual(mean_vals.shape, X_eval.shape)

