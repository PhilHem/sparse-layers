from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from butterfly_layers.layers.custom_linear import CustomLinear


class CustomMLP(nn.Module):
    """MLP that uses CustomLinear layers as drop-in replacements for nn.Linear."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()

        self._validate_dimensions(input_dim, hidden_dims, output_dim)

        hidden_dims = list(hidden_dims)
        layer_dims = [input_dim, *hidden_dims, output_dim]

        modules: list[nn.Module] = []
        for idx in range(len(layer_dims) - 1):
            in_features = layer_dims[idx]
            out_features = layer_dims[idx + 1]

            linear = CustomLinear(in_features, out_features)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("CustomMLP expects a 2D input tensor of shape (batch, features)")

        return self.network(x)

