from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
import torch
from torch import Tensor, nn

from sparse_layers.sse.masking_ops import SSEMaskingOps, SSEMaskingOpsConfig
from sparse_layers.sse.varlen_ops import SSEVarlenOps, SSEVarlenOpsConfig


class SSEAttentionAdaptiveConfig(BaseModel):
    """Configuration for the adaptive SSE attention dispatcher."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Embedding dimension of the input tokens")
    num_partitions: int = Field(
        gt=1, description="Total number of partitions used for sparse execution"
    )
    k: int = Field(
        gt=0, description="Number of partitions selected per token during routing"
    )
    state_rows: int = Field(
        gt=0, description="Rows tracked per partition for compatibility with SSE ops"
    )
    threshold: int = Field(
        gt=0,
        description="Sequence length threshold for switching from masking to varlen ops",
    )
    return_inverse: bool = Field(
        default=False,
        description="Whether to request inverse indices from the varlen implementation",
    )

    @field_validator("k", mode="after")
    @classmethod
    def validate_k_within_partitions(cls, value: int, info: ValidationInfo) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(
                f"k={value} must be <= num_partitions={num_partitions}"
            )
        return value


class SSEAttentionAdaptive(nn.Module):
    """Dispatch between masking and varlen SSE implementations based on sequence length."""

    def __init__(self, config: SSEAttentionAdaptiveConfig) -> None:
        super().__init__()
        self.config = config

        masking_config = SSEMaskingOpsConfig(
            d_model=config.d_model,
            num_partitions=config.num_partitions,
            k=config.k,
        )
        varlen_config = SSEVarlenOpsConfig(
            d_model=config.d_model,
            num_partitions=config.num_partitions,
            k=config.k,
            return_inverse=config.return_inverse,
        )

        self.masking_impl = SSEMaskingOps(masking_config)
        self.varlen_impl = SSEVarlenOps(varlen_config)

        self.threshold = config.threshold

    def forward(
        self, x: Tensor, partition_indices: Tensor
    ) -> Tensor | tuple[Tensor, Tensor, Tensor | None]:
        """Route to masking or varlen implementation depending on sequence length."""

        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.config.d_model:
            raise ValueError(
                f"expected last dimension {self.config.d_model}, received {feature_dim}"
            )

        if partition_indices.dim() != 3:
            raise ValueError(
                "expected partition_indices with shape (batch, seq_len, k)"
            )

        if partition_indices.shape[0] != batch or partition_indices.shape[1] != seq_len:
            raise ValueError(
                "partition_indices must share batch and sequence dimensions with input"
            )

        if partition_indices.shape[2] != self.config.k:
            raise ValueError(
                "expected partition_indices last dimension to equal configured k"
            )

        if partition_indices.dtype != torch.long:
            raise ValueError("partition_indices must have dtype torch.long")

        if seq_len < self.threshold:
            return self.masking_impl(x, partition_indices)

        return self.varlen_impl(x, partition_indices)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.config.d_model}, num_partitions={self.config.num_partitions}, "
            f"k={self.config.k}, threshold={self.threshold}, return_inverse={self.config.return_inverse}"
        )


__all__ = ["SSEAttentionAdaptiveConfig", "SSEAttentionAdaptive"]



