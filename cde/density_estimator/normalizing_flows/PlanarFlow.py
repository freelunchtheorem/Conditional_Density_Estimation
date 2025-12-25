import torch
import torch.nn.functional as F

from .BaseNormalizingFlow import BaseNormalizingFlow


class InvertedPlanarFlow(BaseNormalizingFlow):
    """Planar flow: z' = z + u * tanh(w^T z + b)."""

    def __init__(self, params: torch.Tensor, n_dims: int):
        """
        Args:
            params: tensor of shape (batch_size, 2 * n_dims + 1) containing (u, w, b).
            n_dims: dimension of the target space.
        """
        super().__init__(params, n_dims)
        flow_params = torch.split(params, [n_dims, n_dims, 1], dim=1)
        u = self._handle_input_dimensionality(flow_params[0])
        w = self._handle_input_dimensionality(flow_params[1])
        b = flow_params[2]
        self._u = self._u_circ(u, w)
        self._w = w
        self._b = b

    @staticmethod
    def get_param_size(n_dims: int) -> int:
        return 2 * n_dims + 1

    @staticmethod
    def _u_circ(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        wtu = torch.sum(w * u, dim=1, keepdim=True)
        m_wtu = -1.0 + F.softplus(wtu) + 1e-3
        norm_w_squared = torch.sum(w ** 2, dim=1, keepdim=True)
        return u + (m_wtu - wtu) * (w / norm_w_squared)

    def _wzb(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sum(self._w * z, dim=1, keepdim=True) + self._b

    @staticmethod
    def _der_tanh(z: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.tanh(z) ** 2

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z = self._handle_input_dimensionality(z)
        return z + self._u * torch.tanh(self._wzb(z))

    def ildj(self, z: torch.Tensor) -> torch.Tensor:
        z = self._handle_input_dimensionality(z)
        psi = self._der_tanh(self._wzb(z)) * self._w
        det_grad = 1.0 + torch.sum(self._u * psi, dim=1, keepdim=True)
        return torch.log(torch.abs(det_grad))