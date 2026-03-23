from __future__ import annotations

import torch
from torch import Tensor, nn

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SSEVarlenOpsConfig(BaseModel):
    """Configuration options for the varlen-based SSE implementation."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Embedding dimension of the input tokens")
    num_partitions: int = Field(
        gt=1, description="Total number of partitions maintained for sparse state"
    )
    k: int = Field(
        gt=0, description="Number of partitions selected per token for sparse routing"
    )
    return_inverse: bool = Field(
        default=False,
        description="If True, forward returns mapping tensor to restore original order",
    )

    @field_validator("k")
    @classmethod
    def validate_k_within_partitions(cls, value: int, info: ValidationInfo) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(
                f"k={value} must be <= num_partitions={num_partitions}"
            )
        return value


class SSEVarlenOps(nn.Module):
    """Placeholder for varlen-based sparse sequence execution utilities."""

    def __init__(self, config: SSEVarlenOpsConfig) -> None:
        super().__init__()
        self.config = config

        self.d_model = config.d_model
        self.num_partitions = config.num_partitions
        self.k = config.k
        self.return_inverse = config.return_inverse

    def forward(
        self, x: Tensor, partition_indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Reorder tokens by partition to enable efficient sparse processing.

        Returns:
            packed_tokens: Tensor
            cu_seqlens: Tensor
            inverse_indices: Tensor | None
        """

        self._validate_inputs(x, partition_indices)

        batch, seq_len, _ = x.shape
        device = x.device

        total_partitions = batch * self.num_partitions

        if seq_len == 0:
            empty_tokens = x.new_zeros((0, self.d_model))
            cu_seqlens = torch.zeros(
                total_partitions + 1,
                dtype=torch.int32,
                device=device,
            )
            inverse = (
                torch.full(
                    (batch, 0, self.k),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                if self.return_inverse
                else None
            )
            return empty_tokens, cu_seqlens, inverse

        tokens = (
            x.unsqueeze(2)
            .expand(-1, -1, self.k, -1)
            .contiguous()
            .reshape(batch, -1, self.d_model)
        )
        partitions = partition_indices.reshape(batch, -1)

        time_indices = (
            torch.arange(seq_len, device=device)
            .view(1, seq_len, 1)
            .expand(batch, seq_len, self.k)
            .reshape(batch, -1)
        )
        slot_indices = (
            torch.arange(self.k, device=device)
            .view(1, 1, self.k)
            .expand(batch, seq_len, self.k)
            .reshape(batch, -1)
        )

        cu_seqlens = torch.zeros(
            total_partitions + 1,
            dtype=torch.int32,
            device=device,
        )
        inverse: Tensor | None
        if self.return_inverse:
            inverse = torch.full(
                (batch, seq_len, self.k),
                -1,
                dtype=torch.long,
                device=device,
            )
        else:
            inverse = None

        packed_segments: list[Tensor] = []
        offset = 0
        cu_index = 1

        for b in range(batch):
            partitions_b = partitions[b]
            tokens_b = tokens[b]
            time_b = time_indices[b]
            slot_b = slot_indices[b]

            counts = torch.bincount(
                partitions_b,
                minlength=self.num_partitions,
            )

            for partition in range(self.num_partitions):
                count = int(counts[partition].item())
                if count > 0:
                    mask = partitions_b == partition
                    selected_tokens = tokens_b[mask]
                    packed_segments.append(selected_tokens)

                    if inverse is not None:
                        times = time_b[mask]
                        slots = slot_b[mask]
                        indices = torch.arange(
                            offset,
                            offset + count,
                            device=device,
                            dtype=torch.long,
                        )
                        inverse[b, times, slots] = indices

                    offset += count
                cu_seqlens[cu_index] = offset
                cu_index += 1

        if packed_segments:
            packed_tokens = torch.cat(packed_segments, dim=0)
        else:
            packed_tokens = x.new_zeros((0, self.d_model))

        return packed_tokens, cu_seqlens, inverse

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_partitions={self.num_partitions}, "
            f"k={self.k}, return_inverse={self.return_inverse}"
        )

    def _validate_inputs(self, x: Tensor, partition_indices: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        if partition_indices.dim() != 3:
            raise ValueError(
                "expected partition_indices with shape (batch, seq_len, k)"
            )

        if partition_indices.dtype != torch.long:
            raise ValueError("partition_indices must have dtype torch.long")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, received {feature_dim}"
            )

        if partition_indices.shape[0] != batch or partition_indices.shape[1] != seq_len:
            raise ValueError(
                "partition_indices must share batch and sequence dimensions with input"
            )

        if partition_indices.shape[2] != self.k:
            raise ValueError(
                "expected partition_indices last dimension to equal configured k"
            )

        if partition_indices.numel() > 0:
            min_idx = int(partition_indices.min())
            max_idx = int(partition_indices.max())
            if min_idx < 0 or max_idx >= self.num_partitions:
                raise ValueError(
                    "partition_indices contains values outside valid partition range"
                )


__all__ = ["SSEVarlenOpsConfig", "SSEVarlenOps"]
