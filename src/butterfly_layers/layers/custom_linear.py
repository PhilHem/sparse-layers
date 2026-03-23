from __future__ import annotations

import math

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
