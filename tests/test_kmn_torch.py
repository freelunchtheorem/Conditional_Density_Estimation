import unittest

import numpy as np
import torch

from cde.density_estimator import KernelMixtureNetwork


class TestKernelMixtureNetworkTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def _generate_data(self, n_samples=200):
        X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
        Y = 2.0 * X.squeeze() + np.random.normal(scale=0.1, size=X.shape[0])
        return X, Y.reshape(-1, 1)

    def test_basic_training_pdf_cdf_shapes(self):
        X, Y = self._generate_data()
        model = KernelMixtureNetwork(
            name="kmn_torch_test",
            ndim_x=1,
            ndim_y=1,
            n_centers=10,
            hidden_sizes=(16,),
            hidden_nonlinearity="tanh",
            train_scales=True,
            n_training_epochs=120,
            batch_size=32,
            learning_rate=1e-2,
            data_normalization=True,
            dropout=0.0,
        )
        model.fit(X, Y, verbose=False)

        X_eval = np.zeros((25, 1))
        Y_eval = np.linspace(-1.5, 3.0, 25).reshape(-1, 1)

        pdf_vals = model.pdf(X_eval, Y_eval)
        log_pdf_vals = model.log_pdf(X_eval, Y_eval)
        cdf_vals = model.cdf(X_eval, Y_eval)

        self.assertEqual(pdf_vals.shape, log_pdf_vals.shape)
        self.assertTrue(np.all(pdf_vals >= 0))
        self.assertTrue(np.all(cdf_vals >= 0) and np.all(cdf_vals <= 1))

        sampled = model.sample(X_eval)
        self.assertEqual(sampled[0].shape, X_eval.shape)
        self.assertEqual(sampled[1].shape, X_eval.shape)

        mean_vals = model.mean_(X_eval)
        self.assertEqual(mean_vals.shape, X_eval.shape)

