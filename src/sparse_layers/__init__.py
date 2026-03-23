"""sparse-layers: structured sparse layers for building memory-efficient neural networks."""

from sparse_layers.models import (
    ButterflyMLP,
    ButterflyMultiHeadAttention,
    CustomLinear,
    CustomMLP,
    MultiHeadAttention,
    NaiveSSEMultiHeadAttention,
    SimpleMLP,
    SSEMultiHeadAttention,
    SSEMultiHeadAttentionConfig,
)
from sparse_layers.modules import (
    ButterflyLinear,
    LinearAttention,
    LinearAttentionConfig,
    NaiveMultiPartitionState,
    NaiveSSEAttention,
    PaddedButterflyLinear,
    SSEAttention,
    SSEAttentionAdaptive,
    SSEAttentionAdaptiveConfig,
    SSEAttentionConfig,
    SSEMultiPartitionState,
    SSEMultiPartitionStateConfig,
    SSEPartitionSelector,
    SSEPartitionSelectorConfig,
    SSESparseSoftmax,
    SSESparseSoftmaxConfig,
)
from sparse_layers.ops import (
    SSEMaskingOps,
    SSEMaskingOpsConfig,
    SSEVarlenOps,
    SSEVarlenOpsConfig,
)

__all__ = [
    # Modules
    "ButterflyLinear",
    # Models
    "ButterflyMLP",
    "ButterflyMultiHeadAttention",
    # Baselines (for validation/benchmarking)
    "CustomLinear",
    "CustomMLP",
    "LinearAttention",
    "LinearAttentionConfig",
    "MultiHeadAttention",
    "NaiveMultiPartitionState",
    "NaiveSSEAttention",
    "NaiveSSEMultiHeadAttention",
    "PaddedButterflyLinear",
    "SSEAttention",
    "SSEAttentionAdaptive",
    "SSEAttentionAdaptiveConfig",
    "SSEAttentionConfig",
    # Ops
    "SSEMaskingOps",
    "SSEMaskingOpsConfig",
    "SSEMultiHeadAttention",
    "SSEMultiHeadAttentionConfig",
    "SSEMultiPartitionState",
    "SSEMultiPartitionStateConfig",
    "SSEPartitionSelector",
    "SSEPartitionSelectorConfig",
    "SSESparseSoftmax",
    "SSESparseSoftmaxConfig",
    "SSEVarlenOps",
    "SSEVarlenOpsConfig",
    "SimpleMLP",
]
