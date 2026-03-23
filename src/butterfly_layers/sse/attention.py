from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
import torch
from torch import Tensor, nn

from butterfly_layers.sse.multi_partition_state import SSEMultiPartitionState
from butterfly_layers.sse.multi_partition_state_config import (
    SSEMultiPartitionStateConfig,
)
from butterfly_layers.sse.partition_selector import SSEPartitionSelector
from butterfly_layers.sse.partition_selector_config import SSEPartitionSelectorConfig
from butterfly_layers.sse.sparse_softmax import SSESparseSoftmax
from butterfly_layers.sse.sparse_softmax_config import SSESparseSoftmaxConfig
from butterfly_layers.layers.padded_butterfly_linear import PaddedButterflyLinear


class SSEAttentionConfig(BaseModel):
    """Configuration for both naive and batched SSE attention implementations."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Embedding dimension of the input tokens")
    num_partitions: int = Field(
        gt=1, description="Total number of memory partitions maintained in state"
    )
    k: int = Field(
        gt=0,
        description="Number of partitions selected per token for sparse attention",
    )
    state_rows: int = Field(
        gt=0, description="Number of rows tracked per partition in the sparse state"
    )
    use_butterfly: bool = Field(
        default=False,
        description="Whether to swap dense projections for ButterflyLinear (disabled for Task 3.1)",
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


def _gather_selected_partitions(weights: Tensor, indices: Tensor, state_rows: int) -> Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, state_rows)
    return torch.gather(weights, dim=2, index=gather_index)


def _build_state_config(config: SSEAttentionConfig) -> SSEMultiPartitionStateConfig:
    return SSEMultiPartitionStateConfig(
        num_partitions=config.num_partitions,
        c=config.state_rows,
        d=config.d_model,
    )


class NaiveSSEAttention(nn.Module):
    """Sequential SSE attention placeholder. Implementation follows in later steps."""

    def __init__(self, config: SSEAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_rows = config.state_rows
        self.k = config.k

        selector_config = SSEPartitionSelectorConfig(
            d_model=self.d_model,
            num_partitions=config.num_partitions,
            k=config.k,
        )
        softmax_config = SSESparseSoftmaxConfig(
            num_partitions=config.num_partitions,
            state_rows_per_partition=config.state_rows,
            k=config.k,
        )
        state_config = _build_state_config(config)

        self.partition_selector = SSEPartitionSelector(selector_config)
        self.sparse_softmax = SSESparseSoftmax(softmax_config)
        self.state_mgr = SSEMultiPartitionState(state_config)

        value_layer_cls = PaddedButterflyLinear if config.use_butterfly else nn.Linear
        output_layer_cls = PaddedButterflyLinear if config.use_butterfly else nn.Linear

        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.state_rows)
        self.value = value_layer_cls(self.d_model, self.d_model)
        self.output = output_layer_cls(self.d_model, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, received {feature_dim}"
            )

        if seq_len == 0:
            return torch.zeros(
                batch,
                0,
                self.d_model,
                dtype=x.dtype,
                device=x.device,
            )

        self.reset_state()
        outputs: list[Tensor] = []

        for t in range(seq_len):
            token = x[:, t, :]
            token_seq = token.unsqueeze(1)
            partition_indices = self.partition_selector(token_seq)

            q_t = self.query(token)
            k_t = self.key(token)
            v_t = self.value(token)

            k_sparse = self.sparse_softmax(k_t.unsqueeze(1), partition_indices)
            selected = _gather_selected_partitions(
                k_sparse,
                partition_indices,
                self.state_rows,
            )

            self.state_mgr.update(
                partition_indices,
                selected,
                v_t.unsqueeze(1),
            )

            read = self.state_mgr.read(
                partition_indices,
                q_t.unsqueeze(1),
            )
            outputs.append(read.squeeze(1))

        stacked = torch.stack(outputs, dim=1)
        return self.output(stacked)

    def reset_state(self) -> None:
        self.state_mgr.reset_state()


class SSEAttention(nn.Module):
    """Batched SSE attention placeholder. Implementation follows in later steps."""

    def __init__(self, config: SSEAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_rows = config.state_rows
        self.k = config.k

        selector_config = SSEPartitionSelectorConfig(
            d_model=self.d_model,
            num_partitions=config.num_partitions,
            k=config.k,
        )
        softmax_config = SSESparseSoftmaxConfig(
            num_partitions=config.num_partitions,
            state_rows_per_partition=config.state_rows,
            k=config.k,
        )
        state_config = _build_state_config(config)

        self.partition_selector = SSEPartitionSelector(selector_config)
        self.sparse_softmax = SSESparseSoftmax(softmax_config)
        self.state_mgr = SSEMultiPartitionState(state_config)

        value_layer_cls = PaddedButterflyLinear if config.use_butterfly else nn.Linear
        output_layer_cls = PaddedButterflyLinear if config.use_butterfly else nn.Linear

        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.state_rows)
        self.value = value_layer_cls(self.d_model, self.d_model)
        self.output = output_layer_cls(self.d_model, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, received {feature_dim}"
            )

        if seq_len == 0:
            return torch.zeros(
                batch,
                0,
                self.d_model,
                dtype=x.dtype,
                device=x.device,
            )

        self.reset_state()

        partition_indices = self.partition_selector(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        k_sparse = self.sparse_softmax(k, partition_indices)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            pi_t = partition_indices[:, t : t + 1, :]
            q_t = q[:, t, :].unsqueeze(1)
            v_t = v[:, t, :].unsqueeze(1)
            k_sparse_t = k_sparse[:, t : t + 1, :, :]

            selected = _gather_selected_partitions(
                k_sparse_t,
                pi_t,
                self.state_rows,
            )

            self.state_mgr.update(pi_t, selected, v_t)
            read_t = self.state_mgr.read(pi_t, q_t)
            outputs.append(read_t.squeeze(1))

        stacked = torch.stack(outputs, dim=1)
        return self.output(stacked)

    def reset_state(self) -> None:
        self.state_mgr.reset_state()


__all__ = ["SSEAttentionConfig", "NaiveSSEAttention", "SSEAttention"]


