import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cde.density_estimator.BaseNNMixtureEstimator import BaseNNMixtureEstimator
from cde.utils.center_point_select import sample_center_points

LOG2PI = math.log(2 * math.pi)


def _tanh():
    return nn.Tanh()


def _relu():
    return nn.ReLU()


def _elu():
    return nn.ELU()


def _identity():
    return nn.Identity()


class KernelMixtureNetwork(BaseNNMixtureEstimator):
    """PyTorch implementation of the Kernel Mixture Network."""

    ACTIVATIONS = {
        "tanh": _tanh,
        "relu": _relu,
        "elu": _elu,
        "identity": _identity,
    }

    def __init__(
        self,
        name="KernelMixtureNetwork",
        ndim_x=None,
        ndim_y=None,
        center_sampling_method="k_means",
        n_centers=50,
        keep_edges=True,
        init_scales="default",
        hidden_sizes=(16, 16),
        hidden_nonlinearity="tanh",
        train_scales=True,
        n_training_epochs=1000,
        batch_size=256,
        learning_rate=2e-3,
        x_noise_std=None,
        y_noise_std=None,
        adaptive_noise_fn=None,
        entropy_reg_coef=0.0,
        weight_decay=0.0,
        l2_reg=0.0,
        data_normalization=True,
        dropout=0.0,
        random_seed=None,
        **kwargs,
    ):
        """Initialize the Kernel Mixture Network.

        Args:
            name: estimator name.
            ndim_x: conditioning input dimensionality.
            ndim_y: target dimensionality.
            center_sampling_method: method for selecting mixture centers.
            n_centers: number of kernel centers.
            keep_edges: whether to keep boundary centers.
            init_scales: initial scale specification (defaults to [0.7,0.3]).
            hidden_sizes: sizes of hidden layers.
            hidden_nonlinearity: activation name or module factory.
            train_scales: whether to learn kernel widths.
            n_training_epochs: number of epochs.
            batch_size: minibatch size.
            learning_rate: optimizer learning rate.
            x_noise_std: optional input noise standard deviation.
            y_noise_std: optional target noise standard deviation.
            adaptive_noise_fn: callable returning noise std based on sample count.
            entropy_reg_coef: entropy regularization multiplier.
            weight_decay: L2 weight decay.
            data_normalization: whether to normalize X/Y each fit.
            dropout: dropout probability for hidden layers.
            random_seed: RNG seed for reproducibility.
        """
        super().__init__(
            ndim_x,
            ndim_y,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=n_training_epochs,
            batch_size=batch_size,
        )

        self.name = name
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        self.center_sampling_method = center_sampling_method
        self.keep_edges = keep_edges
        self.entropy_reg_coef = entropy_reg_coef
        self.weight_decay = weight_decay
        self.data_normalization = data_normalization
        self.dropout = dropout
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.adaptive_noise_fn = adaptive_noise_fn
        self.can_sample = True
        self.has_pdf = True
        self.has_cdf = True
        self.l2_reg = kwargs.get("l2_reg", 0.0)

        self.n_centers = n_centers
        self.hidden_sizes = tuple(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.train_scales = train_scales

        if isinstance(init_scales, str) and init_scales == "default":
            init_scales = np.array([0.7, 0.3], dtype=np.float32)
        else:
            init_scales = np.array(init_scales, dtype=np.float32)

        self.n_scales = len(init_scales)
        self.init_scales = init_scales
        self.init_scales_softplus = np.log(np.exp(init_scales) - 1.0)
        self.hidden_activation_spec = hidden_nonlinearity
        self.n_training_epochs = n_training_epochs

        self.register_buffer("_locs_buffer", torch.zeros(0))
        self.log_scales = nn.Parameter(
            torch.tensor(self.init_scales_softplus, dtype=torch.float32), requires_grad=self.train_scales
        )

    def _resolve_activation(self, spec):
        if isinstance(spec, str):
            spec_lower = spec.lower()
            if spec_lower not in self.ACTIVATIONS:
                raise ValueError(f"Unsupported hidden activation '{spec}'")
            return self.ACTIVATIONS[spec_lower]()
        if isinstance(spec, type) and issubclass(spec, nn.Module):
            return spec()
        if callable(spec):
            return spec()
        raise ValueError("hidden_nonlinearity must be a string or nn.Module class")

    def _build_model(self):
        layers = []
        input_dim = self.ndim_x
        for size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(self._resolve_activation(self.hidden_activation_spec))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = size
        layers.append(nn.Linear(input_dim, self.n_centers * self.n_scales))
        return nn.Sequential(*layers)

    def _model_parameters(self):
        return list(self._model.parameters()) + [self.log_scales]

    def _forward(self, x, y):
        return self._model(x)

    def _loss(self, outputs, y):
        logits = outputs
        log_prob = self._log_mixture_density(logits, y)
        loss = -torch.mean(log_prob)
        if self.entropy_reg_coef > 0:
            weights = F.softmax(logits, dim=1)
            entropy = -torch.sum(weights * torch.log(weights + 1e-12), dim=1).mean()
            loss += self.entropy_reg_coef * entropy
        if self.l2_reg > 0:
            penalty_scale = 1.0 + self.epochs / 100.0
            loss += self.l2_reg * penalty_scale * sum((p ** 2).sum() for p in self._model.parameters())
        return loss

    def _component_means_tensor(self):
        locs = self._locs_buffer
        return locs.unsqueeze(1).expand(-1, self.n_scales, -1).reshape(-1, self.ndim_y)

    def _component_scales_tensor(self):
        scales = F.softplus(self.log_scales)
        scale_vec = scales.unsqueeze(1).expand(-1, self.ndim_y)
        return scale_vec.unsqueeze(0).expand(self.n_centers, -1, -1).reshape(-1, self.ndim_y)

    def _maybe_add_noise(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        noise_multiplier = 3.0
        if self.x_noise_std is not None and self.x_noise_std > 0:
            X = X + np.random.normal(scale=noise_multiplier * self.x_noise_std, size=X.shape)
        if self.y_noise_std is not None and self.y_noise_std > 0:
            Y = Y + np.random.normal(scale=noise_multiplier * self.y_noise_std, size=Y.shape)
        return X, Y

    def _component_log_probs(self, y):
        component_means = self._component_means_tensor()
        component_scales = self._component_scales_tensor()
        diff = y.unsqueeze(1) - component_means.unsqueeze(0)
        inv_var = 1.0 / (component_scales ** 2 + 1e-12)
        quadratic = (diff ** 2 * inv_var).sum(dim=-1)
        log_det = torch.log(component_scales).sum(dim=-1)
        const = 0.5 * (self.ndim_y * LOG2PI)
        return -0.5 * quadratic - log_det - const

    def _log_mixture_density(self, logits, y):
        log_weights = F.log_softmax(logits, dim=1)
        component_log_probs = self._component_log_probs(y)
        return torch.logsumexp(log_weights + component_log_probs, dim=1)

    def _normalize_for_eval(self, X, Y):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        Y_norm = self._normalize_array(Y, self.y_mean, self.y_std)
        return (
            torch.from_numpy(X_norm.astype(np.float32)).to(self.device),
            torch.from_numpy(Y_norm.astype(np.float32)).to(self.device),
        )

    def _evaluate_log_pdf(self, X, Y):
        assert self.fitted, "model must be fitted"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        X_tensor, Y_tensor = self._normalize_for_eval(X, Y)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor)
            log_probs = self._log_mixture_density(logits, Y_tensor)
        adjustment = np.sum(np.log(self.y_std + 1e-8))
        return log_probs.cpu().numpy() - adjustment

    def log_pdf(self, X, Y):
        return self._evaluate_log_pdf(X, Y)

    def pdf(self, X, Y):
        return np.exp(self.log_pdf(X, Y))

    def _update_centers(self, Y):
        normalized_Y = self._normalize_array(Y, self.y_mean, self.y_std)
        sampled = sample_center_points(
            normalized_Y,
            method=self.center_sampling_method,
            k=self.n_centers,
            keep_edges=self.keep_edges,
            random_state=self.random_state,
        ).astype(np.float32)
        with torch.no_grad():
            self._locs_buffer.copy_(torch.from_numpy(sampled).to(self.device))

    def fit(self, X, Y, eval_set=None, verbose=True):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
        X, Y = self._maybe_add_noise(X, Y)
        if self.ndim_x is None:
            self.ndim_x = X.shape[1]
        if self.ndim_y is None:
            self.ndim_y = Y.shape[1]
        self._prepare_data(X, Y)
        self._update_centers(Y)
        super().fit(X, Y, verbose=verbose)
        if verbose:
            scales = self._denormalize_scales_numpy(
                self._component_scales_tensor().detach().cpu().numpy()
            )
            print("optimal scales: {}".format(scales[: self.n_scales]))

    def _denormalize_locs_numpy(self, locs):
        return locs * (self.y_std + 1e-8) + self.y_mean

    def _denormalize_scales_numpy(self, scales):
        return scales * (self.y_std + 1e-8)

    def _component_normalized_means(self):
        return self._component_means_tensor()

    def _component_normalized_scales(self):
        return self._component_scales_tensor()

    def _get_mixture_components(self, X):
        X = self._handle_input_dimensionality(X)
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(X_norm.astype(np.float32)).to(self.device))
            weights = F.softmax(logits, dim=1).cpu().numpy()
        locs = self._component_normalized_means().detach().cpu().numpy()
        scales = self._component_normalized_scales().detach().cpu().numpy()
        locs = self._denormalize_locs_numpy(locs)
        scales = self._denormalize_scales_numpy(scales)
        locs = np.tile(locs[None], (X.shape[0], 1, 1))
        scales = np.tile(scales[None], (X.shape[0], 1, 1))
        return weights, locs, scales

    def _param_grid(self):
        return {
        "n_training_epochs": [500, 1000],
        "n_centers": [50, 200],
        "x_noise_std": [0.15, 0.2, 0.3],
            "y_noise_std": [0.1, 0.15, 0.2],
        }

    def __str__(self):
        return (
            f"\nEstimator type: {self.__class__.__name__}\n"
            f" center sampling method: {self.center_sampling_method}\n"
            f" n_centers: {self.n_centers}\n"
            f" keep_edges: {self.keep_edges}\n"
            f" init_scales: {self.init_scales_softplus}\n"
            f" train_scales: {self.train_scales}\n"
            f" n_training_epochs: {self.epochs}\n"
            f" x_noise_std: {self.x_noise_std}\n"
            f" y_noise_std: {self.y_noise_std}\n"
        )

