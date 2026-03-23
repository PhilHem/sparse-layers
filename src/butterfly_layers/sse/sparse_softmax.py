from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from butterfly_layers.sse.sparse_softmax_config import SSESparseSoftmaxConfig


class SSESparseSoftmax(nn.Module):
    """Naive sparse softmax module that will be optimized in later phases."""

    def __init__(self, config: SSESparseSoftmaxConfig) -> None:
        super().__init__()

        self.config = config
        self.num_partitions = config.num_partitions
        self.state_rows_per_partition = config.state_rows_per_partition
        self.k = config.k

    def forward(self, keys: Tensor, partition_indices: Tensor) -> Tensor:
        self._validate_inputs(keys, partition_indices)

        expanded_keys = keys.unsqueeze(2).expand(-1, -1, self.num_partitions, -1)
        partition_scale = torch.linspace(
            1.0,
            1.0 + (self.num_partitions - 1),
            steps=self.num_partitions,
            device=keys.device,
            dtype=keys.dtype,
        )
        scaled_keys = expanded_keys * partition_scale.view(1, 1, self.num_partitions, 1)
        full_softmax = F.softmax(scaled_keys, dim=2)

        mask = keys.new_zeros(keys.shape[0], keys.shape[1], self.num_partitions)
        mask.scatter_(2, partition_indices, 1.0)

        masked = full_softmax * mask.unsqueeze(-1)
        normalization = masked.sum(dim=2, keepdim=True)
        eps = torch.finfo(masked.dtype).eps
        normalized = masked / (normalization + eps)

        return normalized / self.k

    def extra_repr(self) -> str:
        return (
            "num_partitions="
            f"{self.num_partitions}, state_rows_per_partition="
            f"{self.state_rows_per_partition}, k={self.k}"
        )

    def _validate_inputs(self, keys: Tensor, partition_indices: Tensor) -> None:
        if keys.ndim != 3:
            raise ValueError(
                "Expected keys with shape (batch, seq_len, d_model), "
                f"received tensor with {keys.ndim} dimensions"
            )

        if partition_indices.ndim != 3:
            raise ValueError(
                "Expected partition_indices with shape (batch, seq_len, k), "
                f"received tensor with {partition_indices.ndim} dimensions"
            )

        if partition_indices.dtype != torch.long:
            raise ValueError(
                "partition_indices must have dtype torch.long for scatter operations"
            )

        if keys.shape[0] != partition_indices.shape[0]:
            raise ValueError(
                "Batch dimension mismatch: "
                f"keys batch {keys.shape[0]} vs partition_indices batch "
                f"{partition_indices.shape[0]}"
            )

        if keys.shape[1] != partition_indices.shape[1]:
            raise ValueError(
                "Sequence length mismatch: "
                f"keys seq_len {keys.shape[1]} vs partition_indices seq_len "
                f"{partition_indices.shape[1]}"
            )

        if partition_indices.shape[2] != self.k:
            raise ValueError(
                "Expected partition_indices last dimension to equal configured "
                f"k={self.k}, found {partition_indices.shape[2]}"
            )

        if partition_indices.numel() > 0:
            min_index = int(partition_indices.min())
            max_index = int(partition_indices.max())
            if min_index < 0 or max_index >= self.num_partitions:
                raise ValueError(
                    "partition_indices contains values outside the valid range "
                    f"[0, {self.num_partitions - 1}]"
                )


