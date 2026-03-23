from __future__ import annotations

import torch
from torch import Tensor, nn

from sparse_layers.sse.multi_partition_state_config import SSEMultiPartitionStateConfig


class NaiveMultiPartitionState(nn.Module):
    """Naive implementation using Python list of states with explicit loops."""

    def __init__(self, config: SSEMultiPartitionStateConfig) -> None:
        super().__init__()

        self.config = config
        self.num_partitions = config.num_partitions
        self.c = config.c
        self.d = config.d

        # List of separate state matrices
        self.states = nn.ParameterList([
            nn.Parameter(torch.zeros(config.c, config.d), requires_grad=False)
            for _ in range(config.num_partitions)
        ])
        self._forward_deltas: dict[int, Tensor] = {}

    def update(
        self, partition_indices: Tensor, keys: Tensor, values: Tensor
    ) -> None:
        """
        Update state partitions with sparse updates.

        Args:
            partition_indices: (batch, seq_len, k) - which partitions to update
            keys: (batch, seq_len, k, c) - key vectors for selected partitions
            values: (batch, seq_len, d) - value vectors
        """
        self._validate_update_inputs(partition_indices, keys, values)

        # Explicit for-loop over selected partitions
        for b in range(partition_indices.size(0)):
            for t in range(partition_indices.size(1)):
                for i in range(partition_indices.size(2)):
                    partition_idx = partition_indices[b, t, i].item()
                    k_t = keys[b, t, i]  # (c,)
                    v_t = values[b, t]  # (d,)
                    # Update: S_t = S_{t-1} + k^T v
                    outer = torch.outer(k_t, v_t)
                    delta = self._forward_deltas.get(partition_idx)
                    if delta is None:
                        self._forward_deltas[partition_idx] = outer
                    else:
                        self._forward_deltas[partition_idx] = delta + outer
                    self.states[partition_idx].data += outer.detach()

    def _validate_update_inputs(
        self, partition_indices: Tensor, keys: Tensor, values: Tensor
    ) -> None:
        """Validate input shapes for update method."""
        if partition_indices.dim() != 3:
            raise ValueError(
                f"Expected partition_indices with shape (batch, seq_len, k), "
                f"got {partition_indices.shape}"
            )

        if keys.dim() != 4:
            raise ValueError(
                f"Expected keys with shape (batch, seq_len, k, c), "
                f"got {keys.shape}"
            )

        if values.dim() != 3:
            raise ValueError(
                f"Expected values with shape (batch, seq_len, d), "
                f"got {values.shape}"
            )

        batch, seq_len, k = partition_indices.shape
        if keys.shape[:3] != (batch, seq_len, k):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"keys {keys.shape[:2]}"
            )

        if keys.shape[3] != self.c:
            raise ValueError(
                f"Keys last dimension must match c={self.c}, got {keys.shape[3]}"
            )

        if values.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"values {values.shape[:2]}"
            )

        if values.shape[2] != self.d:
            raise ValueError(
                f"Values last dimension must match d={self.d}, got {values.shape[2]}"
            )

    def read(self, partition_indices: Tensor, queries: Tensor) -> Tensor:
        """
        Read from selected partitions.

        Args:
            partition_indices: (batch, seq_len, k) - which partitions to read from
            queries: (batch, seq_len, d) - query vectors

        Returns:
            (batch, seq_len, d) - output vectors
        """
        self._validate_read_inputs(partition_indices, queries)

        batch, seq_len = partition_indices.shape[:2]
        if seq_len == 0:
            return torch.zeros(
                batch, 0, queries.size(-1),
                dtype=queries.dtype,
                device=queries.device
            )

        outputs = []
        for b in range(batch):
            batch_outputs = []
            for t in range(seq_len):
                output_t = torch.zeros(
                    queries.size(-1), dtype=queries.dtype, device=queries.device
                )
                for i in range(partition_indices.size(2)):
                    partition_idx = partition_indices[b, t, i].item()
                    q_t = queries[b, t]  # (d,)
                    state = self.states[partition_idx]  # (c, d)
                    if self._forward_deltas:
                        delta = self._forward_deltas.get(partition_idx)
                        if delta is not None:
                            state = state + (delta - delta.detach())
                    # Read operation: average columns of state matrix weighted by query
                    # This computes: output[j] = mean_i (S[i, j]) * q[j]
                    output_t += state.mean(dim=0) * q_t
                batch_outputs.append(output_t)
            outputs.append(torch.stack(batch_outputs))
        return torch.stack(outputs)

    def _validate_read_inputs(
        self, partition_indices: Tensor, queries: Tensor
    ) -> None:
        """Validate input shapes for read method."""
        if partition_indices.dim() != 3:
            raise ValueError(
                f"Expected partition_indices with shape (batch, seq_len, k), "
                f"got {partition_indices.shape}"
            )

        if queries.dim() != 3:
            raise ValueError(
                f"Expected queries with shape (batch, seq_len, d), "
                f"got {queries.shape}"
            )

        batch, seq_len, k = partition_indices.shape
        if queries.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"queries {queries.shape[:2]}"
            )

        if queries.shape[2] != self.d:
            raise ValueError(
                f"Queries last dimension must match d={self.d}, got {queries.shape[2]}"
            )

    def reset_state(self) -> None:
        """Reset all state partitions to zero."""
        for state in self.states:
            state.data.zero_()
        self._forward_deltas = {}

    def extra_repr(self) -> str:
        return (
            f"num_partitions={self.num_partitions}, "
            f"c={self.c}, d={self.d}"
        )


class SSEMultiPartitionState(nn.Module):
    """Optimized implementation using batched tensor operations."""

    def __init__(self, config: SSEMultiPartitionStateConfig) -> None:
        super().__init__()

        self.config = config
        self.num_partitions = config.num_partitions
        self.c = config.c
        self.d = config.d

        # Single tensor for all partitions: (num_partitions, c, d)
        self.states = nn.Parameter(
            torch.zeros(config.num_partitions, config.c, config.d),
            requires_grad=False
        )
        self._forward_deltas: dict[int, Tensor] = {}

    def update(
        self, partition_indices: Tensor, keys: Tensor, values: Tensor
    ) -> None:
        """
        Update state partitions with sparse updates using batched operations.

        Args:
            partition_indices: (batch, seq_len, k) - which partitions to update
            keys: (batch, seq_len, k, c) - key vectors for selected partitions
            values: (batch, seq_len, d) - value vectors
        """
        self._validate_update_inputs(partition_indices, keys, values)

        batch, seq_len, k = partition_indices.shape

        # Flatten batch and sequence dimensions
        flat_indices = partition_indices.view(-1, k)  # (batch * seq_len, k)
        flat_keys = keys.view(-1, k, self.c)  # (batch * seq_len, k, c)
        flat_values = values.view(-1, self.d)  # (batch * seq_len, d)

        # For each (batch*seq_len, k) entry, compute outer product and update
        # We need to update: states[partition_idx] += outer(key, value)
        for idx in range(batch * seq_len):
            for i in range(k):
                partition_idx = flat_indices[idx, i].item()
                k_t = flat_keys[idx, i]  # (c,)
                v_t = flat_values[idx]  # (d,)
                # Update: states[partition_idx] += outer(k_t, v_t)
                outer = torch.outer(k_t, v_t)
                delta = self._forward_deltas.get(partition_idx)
                if delta is None:
                    self._forward_deltas[partition_idx] = outer
                else:
                    self._forward_deltas[partition_idx] = delta + outer
                self.states.data[partition_idx] += outer.detach()

    def _validate_update_inputs(
        self, partition_indices: Tensor, keys: Tensor, values: Tensor
    ) -> None:
        """Validate input shapes for update method."""
        if partition_indices.dim() != 3:
            raise ValueError(
                f"Expected partition_indices with shape (batch, seq_len, k), "
                f"got {partition_indices.shape}"
            )

        if keys.dim() != 4:
            raise ValueError(
                f"Expected keys with shape (batch, seq_len, k, c), "
                f"got {keys.shape}"
            )

        if values.dim() != 3:
            raise ValueError(
                f"Expected values with shape (batch, seq_len, d), "
                f"got {values.shape}"
            )

        batch, seq_len, k = partition_indices.shape
        if keys.shape[:3] != (batch, seq_len, k):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"keys {keys.shape[:2]}"
            )

        if keys.shape[3] != self.c:
            raise ValueError(
                f"Keys last dimension must match c={self.c}, got {keys.shape[3]}"
            )

        if values.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"values {values.shape[:2]}"
            )

        if values.shape[2] != self.d:
            raise ValueError(
                f"Values last dimension must match d={self.d}, got {values.shape[2]}"
            )

    def read(self, partition_indices: Tensor, queries: Tensor) -> Tensor:
        """
        Read from selected partitions using batched operations.

        Args:
            partition_indices: (batch, seq_len, k) - which partitions to read from
            queries: (batch, seq_len, d) - query vectors

        Returns:
            (batch, seq_len, d) - output vectors
        """
        self._validate_read_inputs(partition_indices, queries)

        batch, seq_len, k = partition_indices.shape

        if seq_len == 0:
            return torch.zeros(
                batch, 0, queries.size(-1),
                dtype=queries.dtype,
                device=queries.device
            )

        # Gather states for selected partitions: (batch, seq_len, k, c, d)
        gathered_states = self.states[partition_indices]  # (batch, seq_len, k, c, d)
        if self._forward_deltas:
            for partition_idx, delta in self._forward_deltas.items():
                adjust = (delta - delta.detach()).view(1, 1, 1, self.c, self.d)
                mask = (partition_indices == partition_idx).unsqueeze(-1).unsqueeze(-1)
                gathered_states = gathered_states + mask.to(gathered_states.dtype) * adjust

        # For each partition, compute: state.mean(dim=3) * query
        # This gives: (batch, seq_len, k, d)
        # Then sum over k partitions: (batch, seq_len, d)
        # For each partition state (c, d), we compute: mean over rows * query
        # Then multiply element-wise with query (d,)
        partition_outputs = gathered_states.mean(dim=3) * queries.unsqueeze(2)  # (batch, seq_len, k, d)
        outputs = partition_outputs.sum(dim=2)  # (batch, seq_len, d)

        return outputs

    def _validate_read_inputs(
        self, partition_indices: Tensor, queries: Tensor
    ) -> None:
        """Validate input shapes for read method."""
        if partition_indices.dim() != 3:
            raise ValueError(
                f"Expected partition_indices with shape (batch, seq_len, k), "
                f"got {partition_indices.shape}"
            )

        if queries.dim() != 3:
            raise ValueError(
                f"Expected queries with shape (batch, seq_len, d), "
                f"got {queries.shape}"
            )

        batch, seq_len, k = partition_indices.shape
        if queries.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"Batch and sequence dimensions must match: "
                f"partition_indices {partition_indices.shape[:2]}, "
                f"queries {queries.shape[:2]}"
            )

        if queries.shape[2] != self.d:
            raise ValueError(
                f"Queries last dimension must match d={self.d}, got {queries.shape[2]}"
            )

    def reset_state(self) -> None:
        """Reset all state partitions to zero."""
        self.states.data.zero_()
        self._forward_deltas = {}

    def extra_repr(self) -> str:
        return (
            f"num_partitions={self.num_partitions}, "
            f"c={self.c}, d={self.d}"
        )
