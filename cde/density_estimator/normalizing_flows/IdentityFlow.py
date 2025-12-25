import torch

from .BaseNormalizingFlow import BaseNormalizingFlow


class IdentityFlow(BaseNormalizingFlow):
    """Identity bijector (y = x)."""

    def __init__(self, params: torch.Tensor, n_dims: int):
        """
        Identity transformation requires no flow parameters.

        Args:
            params: tensor with dimension (batch_size, 0); unused.
            n_dims: dimensionality of the target.
        """
        super().__init__(params, n_dims)

    @staticmethod
    def get_param_size(n_dims: int) -> int:
        return 0

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def ildj(self, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros((y.shape[0], 1), device=y.device)
