import torch


class BaseNormalizingFlow:
    """Torch-friendly base class for individual normalizing flow bijectors."""

    def __init__(self, params: torch.Tensor, n_dims: int):
        """
        Validate the parameter vector shape for the given dimensionality.

        Args:
            params: batched flow parameters (batch_size, param_dim).
            n_dims: dimensionality of the target variable.
        """
        self.n_dims = n_dims
        assert params.shape[1] == self.get_param_size(n_dims), (
            f'Shape is {params.shape[1]}, should be {self.get_param_size(n_dims)}'
        )
        assert params.dim() == 2

    @staticmethod
    def get_param_size(n_dims: int) -> int:
        """Size of the flat parameter vector required for this flow."""
        raise NotImplementedError()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Invert the bijector mapping y -> z."""
        raise NotImplementedError()

    def ildj(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse log-determinant of the Jacobian, i.e., log |det dz/dy|.

        Args:
            y: conditioned tensor of shape (batch_size, n_dims).
        """
        raise NotImplementedError()

    def inverse_and_log_det(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z, log_det) tuple needed for log_prob computations."""
        return self.inverse(y), self.ildj(y)

    @staticmethod
    def _handle_input_dimensionality(z: torch.Tensor) -> torch.Tensor:
        """Ensure the tensor has shape (batch_size, n_dims)."""
        if z.dim() == 1:
            return z.unsqueeze(1)
        return z
