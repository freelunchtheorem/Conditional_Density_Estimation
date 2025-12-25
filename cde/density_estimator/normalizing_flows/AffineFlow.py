import torch

from .BaseNormalizingFlow import BaseNormalizingFlow


class AffineFlow(BaseNormalizingFlow):
    """Affine bijector y = exp(a) * x + b."""

    def __init__(self, params: torch.Tensor, n_dims: int):
        """
        Args:
            params: tensor of shape (batch_size, 2 * n_dims) that encodes log-scale and shift.
            n_dims: dimension of the target space.
        """
        super().__init__(params, n_dims)
        flow_params = torch.split(params, [n_dims, n_dims], dim=1)
        self._a = self._handle_input_dimensionality(flow_params[0])
        self._b = self._handle_input_dimensionality(flow_params[1])

    @staticmethod
    def get_param_size(n_dims: int) -> int:
        return 2 * n_dims

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._b) * torch.exp(-self._a)

    def ildj(self, y: torch.Tensor) -> torch.Tensor:
        return -torch.sum(self._a, dim=1, keepdim=True)
