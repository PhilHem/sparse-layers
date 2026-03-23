"""sparse-layers: structured sparse layers for building memory-efficient neural networks."""

from sparse_layers.models import (
    ButterflyMLP,
    ButterflyMultiHeadAttention,
    CustomMLP,
    MultiHeadAttention,
    NaiveSSEMultiHeadAttention,
    SSEMultiHeadAttention,
    SSEMultiHeadAttentionConfig,
    SimpleMLP,
)
from sparse_layers.modules import (
    ButterflyLinear,
    CustomLinear,
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
    "CustomLinear",
    "LinearAttention",
    "LinearAttentionConfig",
    "NaiveMultiPartitionState",
    "NaiveSSEAttention",
    "PaddedButterflyLinear",
    "SSEAttention",
    "SSEAttentionAdaptive",
    "SSEAttentionAdaptiveConfig",
    "SSEAttentionConfig",
    "SSEMultiPartitionState",
    "SSEMultiPartitionStateConfig",
    "SSEPartitionSelector",
    "SSEPartitionSelectorConfig",
    "SSESparseSoftmax",
    "SSESparseSoftmaxConfig",
    # Ops
    "SSEMaskingOps",
    "SSEMaskingOpsConfig",
    "SSEVarlenOps",
    "SSEVarlenOpsConfig",
    # Models
    "ButterflyMLP",
    "ButterflyMultiHeadAttention",
    "CustomMLP",
    "MultiHeadAttention",
    "NaiveSSEMultiHeadAttention",
    "SSEMultiHeadAttention",
    "SSEMultiHeadAttentionConfig",
    "SimpleMLP",
]
