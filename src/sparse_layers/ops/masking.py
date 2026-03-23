from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from torch import Tensor, nn


class SSEMaskingOpsConfig(BaseModel):
    """Configuration options for the masking-based SSE implementation."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Embedding dimension of the input tokens")
    num_partitions: int = Field(
        gt=1,
        description="Total number of partitions replicated during masking.",
    )
    k: int = Field(gt=0, description="Number of partitions selected per token")

    @field_validator("k")
    @classmethod
    def validate_k_within_partitions(cls, value: int, info: ValidationInfo) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(f"k={value} must be <= num_partitions={num_partitions}")
        return value


class SSEMaskingOps(nn.Module):
    """Masking-based SSE operations for short sequences."""

    def __init__(self, config: SSEMaskingOpsConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.num_partitions = config.num_partitions
        self.k = config.k

    def forward(self, x: Tensor, partition_indices: Tensor) -> Tensor:
        """Replicate activations across partitions and apply binary masks."""

        self._validate_inputs(x, partition_indices)

        batch, seq_len, _ = x.shape
        if seq_len == 0:
            return x.new_zeros(batch, 0, self.num_partitions, self.d_model)

        mask = self._create_mask(partition_indices, x)
        replicated = x.unsqueeze(2).expand(-1, -1, self.num_partitions, -1)

        return replicated * mask.unsqueeze(-1)

    def _validate_inputs(self, x: Tensor, partition_indices: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        if partition_indices.dim() != 3:
            raise ValueError("expected partition_indices with shape (batch, seq_len, k)")

        if partition_indices.dtype != torch.long:
            raise ValueError("partition_indices must have dtype torch.long")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, received {feature_dim}")

        if partition_indices.shape[0] != batch or partition_indices.shape[1] != seq_len:
            raise ValueError(
                "partition_indices must share batch and sequence dimensions with input"
            )

        if partition_indices.shape[2] != self.k:
            raise ValueError("expected partition_indices last dimension to equal configured k")

        if partition_indices.numel() > 0:
            min_idx = int(partition_indices.min())
            max_idx = int(partition_indices.max())
            if min_idx < 0 or max_idx >= self.num_partitions:
                raise ValueError("partition_indices contains values outside valid partition range")

    def _create_mask(self, partition_indices: Tensor, reference: Tensor) -> Tensor:
        batch, seq_len, _ = partition_indices.shape
        mask = reference.new_zeros(batch, seq_len, self.num_partitions)
        if seq_len == 0:
            return mask

        ones = torch.ones_like(partition_indices, dtype=reference.dtype)
        mask.scatter_(2, partition_indices, ones)
        return mask

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_partitions={self.num_partitions}, k={self.k}"


__all__ = ["SSEMaskingOps", "SSEMaskingOpsConfig"]
