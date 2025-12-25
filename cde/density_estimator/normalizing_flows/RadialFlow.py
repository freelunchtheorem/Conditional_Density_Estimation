import torch

from .BaseNormalizingFlow import BaseNormalizingFlow


class InvertedRadialFlow(BaseNormalizingFlow):
    """Radial flow: z' = z + (alpha * beta * (z - gamma)) / (alpha + |z - gamma|)."""

    def __init__(self, params: torch.Tensor, n_dims: int):
        """
        Args:
            params: tensor of shape (batch_size, n_dims + 2) encoding (alpha, beta, gamma).
            n_dims: dimension of the target space.
        """
        super().__init__(params, n_dims)
        flow_params = torch.split(params, [1, 1, n_dims], dim=1)
        self._alpha = self._alpha_circ(self._handle_input_dimensionality(flow_params[0]))
        self._beta = self._beta_circ(self._handle_input_dimensionality(flow_params[1]))
        self._gamma = self._handle_input_dimensionality(flow_params[2])

    @staticmethod
    def get_param_size(n_dims: int) -> int:
        return 2 + n_dims

    def _r(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(z - self._gamma), dim=1, keepdim=True)

    def _h(self, r: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self._alpha + r)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z = self._handle_input_dimensionality(z)
        r = self._r(z)
        h = self._h(r)
        return z + (self._alpha * self._beta * h) * (z - self._gamma)

    def ildj(self, z: torch.Tensor) -> torch.Tensor:
        z = self._handle_input_dimensionality(z)
        r = self._r(z)
        h = self._h(r)
        der_h = -1.0 / (self._alpha + r) ** 2
        ab = self._alpha * self._beta
        det = (1.0 + ab * h) ** (self.n_dims - 1) * (1.0 + ab * h + ab * der_h * r)
        return torch.log(det)

    @staticmethod
    def _alpha_circ(alpha: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(alpha)

    @staticmethod
    def _beta_circ(beta: torch.Tensor) -> torch.Tensor:
        return torch.exp(beta) - 1.0
