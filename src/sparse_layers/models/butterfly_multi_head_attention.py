from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sparse_layers.modules.butterfly_linear import ButterflyLinear
from sparse_layers.ops.butterfly import _is_power_of_two


class ButterflyMultiHeadAttention(nn.Module):
    """Multi-head self-attention using ButterflyLinear projections."""

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

        if not _is_power_of_two(d_model):
            raise ValueError("ButterflyMultiHeadAttention requires d_model to be a power of two")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = ButterflyLinear(d_model, d_model)
        self.key = ButterflyLinear(d_model, d_model)
        self.value = ButterflyLinear(d_model, d_model)
        self.out = ButterflyLinear(d_model, d_model)
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


__all__ = ["ButterflyMultiHeadAttention"]
