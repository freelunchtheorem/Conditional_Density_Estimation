import logging
import pickle
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.serialization import add_safe_globals

from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator

logger = logging.getLogger(__name__)


class BaseNNEstimator(BaseDensityEstimator, nn.Module):
    """PyTorch base estimator with training loop, normalization, and optimizer scaffolding."""

    data_normalization = False
    x_noise_std = 0.0
    y_noise_std = 0.0
    dropout = 0.0

    def __init__(
        self,
        ndim_x: int,
        ndim_y: int,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler_cls: Optional[Type[lr_scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        BaseDensityEstimator.__init__(self)

        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}

        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[lr_scheduler._LRScheduler] = None

        self.fitted = False
        self.data_statistics = {}

    def reset_fit(self):
        """Reset internal modules so the estimator can be re-trained."""
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self.fitted = False

    def _build_model(self) -> nn.Module:
        """Override in subclasses to construct the nn.Module architecture."""
        raise NotImplementedError()

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes tensors needed for pdf/logpdf evaluation."""
        raise NotImplementedError()

    def _loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss used during training."""
        raise NotImplementedError()

    def _model_parameters(self):
        return list(self._model.parameters())

    def _ensure_model(self):
        if self._model is None:
            self._model = self._build_model()
            self._model = self._model.to(self.device)
            self.to(self.device)
            optimizer_params = self._model_parameters()
            self._optimizer = self.optimizer_cls(
                optimizer_params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
            if self.scheduler_cls is not None and self.scheduler_kwargs is not None:
                self._scheduler = self.scheduler_cls(self._optimizer, **self.scheduler_kwargs)

    def _normalize_array(self, array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        if not self.data_normalization:
            return array
        std_safe = std + 1e-8
        return (array - mean) / std_safe

    def _denormalize_array(self, array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        if not self.data_normalization:
            return array
        return array * (std + 1e-8) + mean

    def _prepare_data(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 2 and Y.ndim == 2
        if self.data_normalization:
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)
            self.y_mean = Y.mean(axis=0)
            self.y_std = Y.std(axis=0)
        else:
            self.x_mean = np.zeros(self.ndim_x, dtype=np.float32)
            self.x_std = np.ones(self.ndim_x, dtype=np.float32)
            self.y_mean = np.zeros(self.ndim_y, dtype=np.float32)
            self.y_std = np.ones(self.ndim_y, dtype=np.float32)

        self.data_statistics = {
            "X_mean": self.x_mean,
            "X_std": self.x_std,
            "Y_mean": self.y_mean,
            "Y_std": self.y_std,
        }

        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        Y_norm = self._normalize_array(Y, self.y_mean, self.y_std)

        return (
            torch.from_numpy(X_norm.astype(np.float32)).to(self.device),
            torch.from_numpy(Y_norm.astype(np.float32)).to(self.device),
        )

    def _normalize_XY(self, X: np.ndarray, Y: np.ndarray):
        X_norm = self._normalize_array(X, self.x_mean, self.x_std)
        Y_norm = self._normalize_array(Y, self.y_mean, self.y_std)
        return X_norm, Y_norm

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
        self._ensure_model()
        X_tensor, Y_tensor = self._prepare_data(X, Y)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                self._optimizer.zero_grad()
                outputs = self._forward(x_batch, y_batch)
                loss = self._loss(outputs, y_batch)
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss)
            if self._scheduler:
                self._scheduler.step()
            if verbose and epoch % max(1, self.epochs // 10) == 0:
                avg_loss = epoch_loss / len(loader)
                logger.info("Epoch %d/%d training loss %.4f", epoch, self.epochs, avg_loss)

        self.fitted = True

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean(self.log_pdf(X, Y)))

    def _add_entropy_regularization(self, loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        coef = getattr(self, "entropy_reg_coef", 0.0)
        if coef <= 0:
            return loss
        weights = F.softmax(logits, dim=1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-12), dim=1).mean()
        return loss + coef * entropy

    def _add_l1_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        coef = getattr(self, "l1_reg", 0.0)
        if coef <= 0:
            return loss
        penalty = sum(p.abs().sum() for p in self._model.parameters())
        return loss + coef * penalty

    def _add_l2_regularization(self, loss: torch.Tensor, penalty_scale: float = 1.0) -> torch.Tensor:
        coef = getattr(self, "l2_reg", 0.0)
        if coef <= 0:
            return loss
        penalty = sum((p ** 2).sum() for p in self._model.parameters())
        return loss + coef * penalty_scale * penalty

    def pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement pdf evaluation.")

    def log_pdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement log_pdf evaluation.")

    def save_state(self, path: str) -> None:
        """Persist model state and dataset statistics to disk."""
        self._ensure_model()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "data_statistics": self.data_statistics,
        }
        torch.save(payload, target)

    def load_state(self, path: str, map_location: Optional[torch.device] = None) -> None:
        """Load previously saved weights and statistics."""
        self._ensure_model()
        target = Path(path)
        try:
            checkpoint = torch.load(target, map_location=(map_location or self.device), weights_only=True)
        except (RuntimeError, pickle.UnpicklingError) as exc:
            add_safe_globals([np.core.multiarray._reconstruct])
            checkpoint = torch.load(target, map_location=(map_location or self.device), weights_only=False)
        state_dict = checkpoint["state_dict"]
        locs_buffer = state_dict.get("_locs_buffer")
        if locs_buffer is not None:
            self._locs_buffer = torch.zeros_like(locs_buffer, device=self.device)
        self.load_state_dict(state_dict)
        data_statistics = checkpoint.get("data_statistics", {})
        self.data_statistics = data_statistics
        for key, value in data_statistics.items():
            setattr(self, key, value)
            lower_key = key.lower()
            if lower_key != key:
                setattr(self, lower_key, value)
        self.fitted = True
