"""Composed architectures built from sparse modules."""

from sparse_layers.models._baselines import CustomLinear, CustomMLP, MultiHeadAttention, SimpleMLP
from sparse_layers.models.butterfly_mlp import ButterflyMLP
from sparse_layers.models.butterfly_multi_head_attention import (
    ButterflyMultiHeadAttention,
)
from sparse_layers.models.sse_multi_head_attention import (
    NaiveSSEMultiHeadAttention,
    SSEMultiHeadAttention,
    SSEMultiHeadAttentionConfig,
)

__all__ = [
    # Public API
    "ButterflyMLP",
    "ButterflyMultiHeadAttention",
    # Baselines (for validation/benchmarking)
    "CustomLinear",
    "CustomMLP",
    "MultiHeadAttention",
    "NaiveSSEMultiHeadAttention",
    "SSEMultiHeadAttention",
    "SSEMultiHeadAttentionConfig",
    "SimpleMLP",
]
