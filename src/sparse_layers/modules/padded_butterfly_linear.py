from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from sparse_layers.modules.butterfly_linear import ButterflyLinear
from sparse_layers.ops.butterfly import _next_power_of_two


class PaddedButterflyLinear(nn.Module):
    """ButterflyLinear wrapper supporting arbitrary input and output dimensions.

    The module pads the input features to the next power-of-two dimension,
    applies a square :class:`ButterflyLinear` transformation, and slices the
    result back to the requested output dimensionality.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer")

        self.in_features = in_features
        self.out_features = out_features
        self._padded_dim = _next_power_of_two(max(in_features, out_features))
        self._input_padding = self._padded_dim - in_features
        self._output_slice = slice(0, out_features)

        self.inner = ButterflyLinear(self._padded_dim, self._padded_dim, bias=bias)

    @property
    def bias(self) -> nn.Parameter | None:
        return self.inner.bias

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dimension {self.in_features}, got {input.shape[-1]}")

        original_shape = input.shape[:-1]
        x = input.reshape(-1, self.in_features)

        if self._input_padding > 0:
            x = F.pad(x, (0, self._input_padding))

        x = self.inner(x)
        x = x[..., self._output_slice]

        return x.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"padded_dim={self._padded_dim}, bias={self.bias is not None}"
        )


__all__ = ["PaddedButterflyLinear"]
