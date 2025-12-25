import math
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def _mdn_tanh():
    return nn.Tanh()


def _mdn_relu():
    return nn.ReLU()


def _mdn_elu():
    return nn.ELU()


def _mdn_identity():
    return nn.Identity()

from cde.density_estimator.BaseNNMixtureEstimator import BaseNNMixtureEstimator


class MixtureDensityNetwork(BaseNNMixtureEstimator):
    """Torch-based Mixture Density Network estimator."""

    ACTIVATIONS = {
        "tanh": _mdn_tanh,
        "relu": _mdn_relu,
        "elu": _mdn_elu,
        "identity": _mdn_identity,
    }

    def __init__(
        self,
        name: str = "MixtureDensityNetwork",
        ndim_x: Optional[int] = None,
        ndim_y: Optional[int] = None,
        n_centers: int = 10,
        hidden_sizes: Sequence[int] = (16, 16),
        hidden_nonlinearity: Union[str, Callable[[], nn.Module]] = "tanh",
        n_training_epochs: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        x_noise_std: Optional[float] = None,
        y_noise_std: Optional[float] = None,
        adaptive_noise_fn: Optional[Callable[[int, int], float]] = None,
        entropy_reg_coef: float = 0.0,
        weight_decay: float = 0.0,
        weight_normalization: bool = True,
        data_normalization: bool = True,
        dropout: float = 0.0,
        l2_reg: float = 0.0,
        l1_reg: float = 0.0,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize the MDN.

        Args:
            name: estimator name.
            ndim_x: dimensionality of the conditioning input.
            ndim_y: dimensionality of the target.
            n_centers: number of mixture components.
            hidden_sizes: sizes of the hidden layers.
            hidden_nonlinearity: activation constructor or name.
            n_training_epochs: training epochs.
            batch_size: minibatch size.
            learning_rate: optimizer learning rate.
            x_noise_std: optional standard deviation for input noise.
            y_noise_std: optional standard deviation for target noise.
            adaptive_noise_fn: callable returning std based on data size.
            entropy_reg_coef: entropy regularization coefficient.
            weight_decay: optimizer weight decay.
            weight_normalization: whether to apply weight normalization.
            data_normalization: enable normalization of X/Y.
            dropout: dropout probability.
            l2_reg: l2 penalty for custom losses.
            l1_reg: l1 penalty for custom losses.
            random_seed: seed for deterministic behavior.
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
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.random_state = np.random.RandomState(seed=random_seed)

        self.n_centers = n_centers
        self.hidden_sizes = tuple(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.n_training_epochs = n_training_epochs
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        self.data_normalization = data_normalization
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.adaptive_noise_fn = adaptive_noise_fn
        self.entropy_reg_coef = entropy_reg_coef
        self.weight_normalization = weight_normalization
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg

        self.can_sample = True
        self.has_pdf = True
        self.has_cdf = True

        self.hidden_activation_factory = self._resolve_activation(hidden_nonlinearity)

    def _resolve_activation(self, spec: Union[str, Callable[[], nn.Module]]) -> Callable[[], nn.Module]:
        if isinstance(spec, str):
            spec_lower = spec.lower()
            if spec_lower not in self.ACTIVATIONS:
                raise ValueError(f"Unsupported activation '{spec}'")
            return self.ACTIVATIONS[spec_lower]
        if isinstance(spec, type) and issubclass(spec, nn.Module):
            return lambda: spec()
        if callable(spec):
            return lambda: spec()
        raise ValueError("hidden_nonlinearity must be a string or callable returning nn.Module")

    def _linear(self, in_features: int, out_features: int) -> nn.Module:
        linear = nn.Linear(in_features, out_features)
        if self.weight_normalization:
            return nn.utils.weight_norm(linear)
        return linear

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []
        input_dim = self.ndim_x
        for size in self.hidden_sizes:
            layers.append(self._linear(input_dim, size))
            layers.append(self.hidden_activation_factory())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = size
        output_dim = self.n_centers * (2 * self.ndim_y) + self.n_centers
        layers.append(self._linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _split_outputs(
        self, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        locs_end = self.n_centers * self.ndim_y
        scales_end = locs_end + self.n_centers * self.ndim_y
        logits = outputs[:, scales_end:]
        locs = outputs[:, :locs_end].reshape(-1, self.n_centers, self.ndim_y)
        scales = outputs[:, locs_end:scales_end].reshape(-1, self.n_centers, self.ndim_y)
        scales = F.softplus(scales) + 1e-8
        return logits, locs, scales

    def _component_log_probs(
        self, locs: torch.Tensor, scales: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        diff = y.unsqueeze(1) - locs
        inv_var = 1.0 / (scales ** 2 + 1e-12)
        quadratic = (diff ** 2 * inv_var).sum(dim=-1)
        log_det = torch.log(scales).sum(dim=-1)
        const = 0.5 * (self.ndim_y * LOG2PI)
        return -0.5 * quadratic - log_det - const

    def _log_mixture_density(
        self, logits: torch.Tensor, locs: torch.Tensor, scales: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        component_log_probs = self._component_log_probs(locs, scales, y)
        log_weights = F.log_softmax(logits, dim=1)
        return torch.logsumexp(log_weights + component_log_probs, dim=1)

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def _loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, locs, scales = self._split_outputs(outputs)
        log_prob = self._log_mixture_density(logits, locs, scales, y)
        loss = -torch.mean(log_prob)
        loss = self._add_entropy_regularization(loss, logits)
        loss = self._add_l1_regularization(loss)
        loss = self._add_l2_regularization(loss)
        return loss


    def _maybe_add_noise(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        noise_std = None
        if self.adaptive_noise_fn is not None:
            noise_std = float(self.adaptive_noise_fn(X.shape[0], self.ndim_x))
        x_noise = noise_std if noise_std is not None else self.x_noise_std
        y_noise = noise_std if noise_std is not None else self.y_noise_std
        if x_noise is not None and x_noise > 0:
            X = X + self.random_state.normal(scale=x_noise, size=X.shape)
        if y_noise is not None and y_noise > 0:
            Y = Y + self.random_state.normal(scale=y_noise, size=Y.shape)
        return X, Y

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        verbose: bool = False,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
        if self.ndim_x is None:
            self.ndim_x = X.shape[1]
        if self.ndim_y is None:
            self.ndim_y = Y.shape[1]
        if eval_set is not None:
            tuple(self._handle_input_dimensionality(*eval_set))
        X, Y = self._maybe_add_noise(X, Y)
        super().fit(X, Y, verbose=verbose)

    def _normalize_for_eval(self, X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        Y_norm = self._normalize_array(Y, self.y_mean, self.y_std)
        return (
            torch.from_numpy(X_norm.astype(np.float32)).to(self.device),
            torch.from_numpy(Y_norm.astype(np.float32)).to(self.device),
        )

    def _evaluate_log_pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        assert self.fitted, "model must be fitted"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        X_tensor, Y_tensor = self._normalize_for_eval(X, Y)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            logits, locs, scales = self._split_outputs(outputs)
            log_probs = self._log_mixture_density(logits, locs, scales, Y_tensor)
        adjustment = np.sum(np.log(self.y_std + 1e-8))
        return log_probs.detach().cpu().numpy() - adjustment

    def log_pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self._evaluate_log_pdf(X, Y)

    def pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(X, Y))

    def _get_mixture_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.fitted
        X = self._handle_input_dimensionality(X, fitting=False)
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        X_tensor = torch.from_numpy(X_norm.astype(np.float32)).to(self.device)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            logits, locs, scales = self._split_outputs(outputs)
            weights = F.softmax(logits, dim=1).cpu().numpy()
            locs = locs.cpu().numpy()
            scales = scales.cpu().numpy()
        locs = locs * (self.y_std + 1e-8) + self.y_mean
        scales = scales * (self.y_std + 1e-8)
        return weights, locs, scales

    def _param_grid(self) -> dict[str, Iterable]:
        return {
            "n_training_epochs": [500, 1000],
            "n_centers": [5, 10, 20],
            "x_noise_std": [0.1, 0.15, 0.2, 0.3],
            "y_noise_std": [0.1, 0.15, 0.2],
        }

    def __str__(self) -> str:
        return (
            f"\nEstimator type: {self.__class__.__name__}\n"
            f" n_centers: {self.n_centers}\n"
            f" entropy_reg_coef: {self.entropy_reg_coef}\n"
            f" data_normalization: {self.data_normalization}\n"
            f" weight_normalization: {self.weight_normalization}\n"
            f" n_training_epochs: {self.n_training_epochs}\n"
            f" x_noise_std: {self.x_noise_std}\n"
            f" y_noise_std: {self.y_noise_std}\n"
        )