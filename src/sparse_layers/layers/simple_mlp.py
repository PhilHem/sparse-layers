from collections.abc import Sequence
import math

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """A minimal configurable multi-layer perceptron for testing."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()

        self._validate_dimensions(input_dim, hidden_dims, output_dim)

        hidden_dims = list(hidden_dims)
        layer_dims = [input_dim, *hidden_dims, output_dim]

        modules: list[nn.Module] = []
        for idx in range(len(layer_dims) - 1):
            in_features = layer_dims[idx]
            out_features = layer_dims[idx + 1]

            linear = nn.Linear(in_features, out_features)
            self._initialize_linear(linear)
            modules.append(linear)

            if idx < len(layer_dims) - 2:
                modules.append(nn.ReLU())

        self.network = nn.Sequential(*modules)

    @staticmethod
    def _validate_dimensions(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one positive integer")

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers")

        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")

    @staticmethod
    def _initialize_linear(layer: nn.Linear) -> None:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if layer.bias is not None:
            fan_in = layer.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("SimpleMLP expects a 2D input tensor of shape (batch, features)")

        return self.network(x)

