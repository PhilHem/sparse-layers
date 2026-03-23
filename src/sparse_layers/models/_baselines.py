"""Baseline implementations for validation and benchmarking.

These are standard (non-sparse) reference implementations used to verify
that sparse variants produce equivalent outputs and to measure speedups.
Not part of the public API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class CustomLinear(nn.Module):
    """Drop-in replacement for nn.Linear with identical behavior."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer")

        self.in_features = in_features
        self.out_features = out_features

        self.weight: Parameter = Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias: Parameter | None = Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


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


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer inspired by 'Attention Is All You Need'."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._scaling = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")
        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, received {feature_dim}")

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)

        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self._scaling

        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError("mask shape must match (batch, seq_len)")
            if mask.dtype != torch.bool:
                raise ValueError("mask must have dtype torch.bool")
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(expanded_mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, value_heads)
        merged_context = self._merge_heads(context)

        output = self.out(merged_context)
        return output

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        reshaped = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return reshaped.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        transposed = tensor.permute(0, 2, 1, 3)
        return transposed.reshape(batch_size, seq_len, self.d_model)
