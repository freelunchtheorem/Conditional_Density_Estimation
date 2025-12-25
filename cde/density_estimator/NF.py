import numpy as np
from typing import Any, Callable, Iterable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import DataLoader, TensorDataset

from .BaseNNEstimator import BaseNNEstimator
from .normalizing_flows import FLOWS


class NormalizingFlowEstimator(BaseNNEstimator):
    """PyTorch port of the original normalizing flow estimator."""

    ACTIVATIONS = {
        "tanh": lambda: nn.Tanh(),
        "relu": lambda: nn.ReLU(),
        "elu": lambda: nn.ELU(),
        "identity": lambda: nn.Identity(),
    }

    def __init__(
        self,
        name: str,
        ndim_x: int,
        ndim_y: int,
        flows_type: Tuple[str, ...] | None = None,
        n_flows: int = 10,
        hidden_sizes: Sequence[int] = (16, 16),
        hidden_nonlinearity: Union[str, Callable[[], nn.Module]] = "tanh",
        n_training_epochs: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-2,
        x_noise_std: float | None = None,
        y_noise_std: float | None = None,
        adaptive_noise_fn: Callable[[int, int], float] | None = None,
        weight_decay: float = 0.0,
        weight_normalization: bool = True,
        data_normalization: bool = True,
        dropout: float = 0.0,
        l2_reg: float = 0.0,
        l1_reg: float = 0.0,
        random_seed: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            name: estimator name (used for logging / serialization).
            ndim_x: dimensionality of the conditioning input.
            ndim_y: dimensionality of the target variable.
            flows_type: tuple of flow identifiers defining the chain.
            n_flows: fallback count of radial flows if flows_type is None.
            hidden_sizes: sizes of the hidden MLP layers.
            hidden_nonlinearity: activation key or constructor.
            n_training_epochs: number of training epochs.
            batch_size: minibatch size used during training.
            learning_rate: optimizer learning rate.
            x_noise_std/y_noise_std: optional additive noise.
            adaptive_noise_fn: callable returning adaptive noise std.
            weight_decay: optimizer L2 weight decay.
            weight_normalization: applies weight normalization if True.
            data_normalization: z-score normalizes X/Y before training.
            dropout: dropout probability used inside the MLP.
            l2_reg/l1_reg: additional penalties applied inside `_loss`.
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
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.random_state = np.random.RandomState(seed=random_seed)

        if flows_type is None:
            flows_type = ("affine",) + ("radial",) * n_flows
        assert all(flow in FLOWS for flow in flows_type)
        self.flows_type = flows_type
        self.flow_classes = [FLOWS[flow_name] for flow_name in flows_type]
        self._param_split_sizes = [
            flow.get_param_size(self.ndim_y) for flow in self.flow_classes
        ]

        self.hidden_sizes = tuple(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_activation_factory = self._resolve_activation(hidden_nonlinearity)

        self.n_training_epochs = n_training_epochs
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.adaptive_noise_fn = adaptive_noise_fn
        self.weight_normalization = weight_normalization
        self.data_normalization = data_normalization
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg

        self.gradient_clipping = "planar" in flows_type

        self.can_sample = False
        self.has_pdf = True
        self.has_cdf = self.ndim_y == 1
        self.fitted = False

        self._ensure_model()

    def _resolve_activation(
        self, spec: Union[str, Callable[[], nn.Module]]
    ) -> Callable[[], nn.Module]:
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
        output_dim = sum(self._param_split_sizes)
        layers.append(self._linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def _build_flows(self, outputs: torch.Tensor):
        splits = torch.split(outputs, self._param_split_sizes, dim=1)
        return [
            flow_class(params, self.ndim_y)
            for flow_class, params in zip(self.flow_classes, splits)
        ]

    def _inverse_flows(self, y: torch.Tensor, flows):
        current = y
        total_log_det = torch.zeros((y.shape[0], 1), device=y.device)
        for flow in reversed(flows):
            current, log_det = flow.inverse_and_log_det(current)
            total_log_det = total_log_det + log_det
        return current, total_log_det

    def _base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.ndim_y == 1:
            dist = Normal(
                torch.tensor(0.0, device=z.device),
                torch.tensor(1.0, device=z.device),
            )
            return dist.log_prob(z.squeeze(-1))
        mean = torch.zeros(self.ndim_y, device=z.device)
        cov = torch.eye(self.ndim_y, device=z.device)
        dist = MultivariateNormal(mean, covariance_matrix=cov)
        return dist.log_prob(z)

    def _flow_log_prob(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        flows = self._build_flows(outputs)
        base_z, log_det = self._inverse_flows(y, flows)
        log_det = log_det.squeeze(-1)
        return self._base_log_prob(base_z) + log_det

    def _loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_prob = self._flow_log_prob(outputs, y)
        loss = -torch.mean(log_prob)
        if self.l1_reg > 0:
            loss = loss + self.l1_reg * sum(p.abs().sum() for p in self._model.parameters())
        if self.l2_reg > 0:
            loss = loss + self.l2_reg * sum((p ** 2).sum() for p in self._model.parameters())
        return loss

    def _maybe_add_noise(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False, eval_set=None, **kwargs: Any):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
        if eval_set is not None:
            tuple(self._handle_input_dimensionality(*eval_set))
        X, Y = self._maybe_add_noise(X, Y)
        X_tensor, Y_tensor = self._prepare_data(X, Y)
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self._ensure_model()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                self._optimizer.zero_grad()
                outputs = self._model(x_batch)
                loss = self._loss(outputs, y_batch)
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 3e5)
                self._optimizer.step()
                epoch_loss += float(loss)
            if self._scheduler is not None:
                self._scheduler.step()
            if verbose and epoch % max(1, self.epochs // 10) == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"Epoch {epoch}/{self.epochs} training loss {avg_loss:.4f}")

        self.fitted = True

    def _normalize_for_eval(self, X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        Y_norm = self._normalize_array(Y, self.y_mean, self.y_std)
        return (
            torch.from_numpy(X_norm.astype(np.float32)).to(self.device),
            torch.from_numpy(Y_norm.astype(np.float32)).to(self.device),
        )

    def _evaluate_log_pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        assert self.fitted, "model must be fitted"
        X_tensor, Y_tensor = self._normalize_for_eval(X, Y)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            log_probs = self._flow_log_prob(outputs, Y_tensor)
        adjustment = np.sum(np.log(self.y_std + 1e-8))
        return log_probs.cpu().numpy() - adjustment

    def log_pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self._evaluate_log_pdf(X, Y)

    def pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(X, Y))

    def cdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        assert self.has_cdf, "CDF implemented only for 1-D outputs"
        X_tensor, Y_tensor = self._normalize_for_eval(X, Y)
        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            flows = self._build_flows(outputs)
            base_z, _ = self._inverse_flows(Y_tensor, flows)
        dist = Normal(0.0, 1.0)
        cdf_vals = dist.cdf(base_z.squeeze(-1))
        return cdf_vals.cpu().numpy()

    def _param_grid(self) -> dict[str, Iterable]:
        return {
            "n_training_epochs": [500, 1000, 1500],
            "hidden_sizes": [(16, 16), (32, 32)],
            "flows_type": [
                ("affine", "radial", "radial", "radial"),
                ("affine", "radial", "radial", "radial", "radial"),
                ("affine", "radial", "radial", "radial", "radial", "radial"),
                ("planar", "planar", "planar"),
                ("affine", "planar", "planar", "planar"),
                ("affine", "planar", "planar", "planar", "planar"),
                ("affine", "radial", "planar", "radial", "planar"),
                ("affine", "radial", "planar", "radial", "planar", "radial"),
            ],
            "x_noise_std": [0.1, 0.2, 0.4, None],
            "y_noise_std": [0.01, 0.02, 0.05, 0.1, 0.2, None],
            "weight_decay": [1e-5, 0.0],
        }

    def __str__(self) -> str:
        return (
            f"\nEstimator type: {self.__class__.__name__}\n"
            f" flows_type: {self.flows_type}\n"
            f" data_normalization: {self.data_normalization}\n"
            f" weight_normalization: {self.weight_normalization}\n"
            f" n_training_epochs: {self.n_training_epochs}\n"
            f" x_noise_std: {self.x_noise_std}\n"
            f" y_noise_std: {self.y_noise_std}\n"
        )
