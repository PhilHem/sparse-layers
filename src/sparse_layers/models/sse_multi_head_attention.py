from __future__ import annotations

from typing import Final, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torch import Tensor, nn

from sparse_layers.modules.padded_butterfly_linear import PaddedButterflyLinear
from sparse_layers.modules.sse_attention import SSEAttention, SSEAttentionConfig


class SSEMultiHeadAttentionConfig(BaseModel):
    """Configuration shared by naive and batched SSE multi-head attention."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Input feature dimension")
    num_heads: int = Field(gt=0, description="Number of parallel attention heads")
    num_partitions: int = Field(gt=1, description="Total number of sparse state partitions")
    k: int = Field(
        gt=0,
        description="Number of partitions selected per token for sparse attention",
    )
    state_rows: int = Field(gt=0, description="Rows per partition maintained by the sparse state")
    use_butterfly: bool = Field(
        default=False,
        description="Swap dense projections for ButterflyLinear (handled via padding).",
    )

    @field_validator("k", mode="after")
    @classmethod
    def validate_k_within_partitions(cls, value: int, info: ValidationInfo) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(f"k={value} must be <= num_partitions={num_partitions}")
        return value

    @model_validator(mode="after")
    def validate_divisible_d_model(self) -> SSEMultiHeadAttentionConfig:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
            )
        return self


class NaiveSSEMultiHeadAttention(nn.Module):
    """Reference implementation using one SSE attention module per head."""

    def __init__(self, config: SSEMultiHeadAttentionConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model: Final[int] = config.d_model
        self.num_heads: Final[int] = config.num_heads
        self.head_dim: Final[int] = self.d_model // self.num_heads

        attention_config = SSEAttentionConfig(
            d_model=self.head_dim,
            num_partitions=config.num_partitions,
            k=config.k,
            state_rows=config.state_rows,
            use_butterfly=config.use_butterfly,
        )

        self.heads = nn.ModuleList([SSEAttention(attention_config) for _ in range(self.num_heads)])

        output_layer_cls = PaddedButterflyLinear if config.use_butterfly else nn.Linear
        self.output = output_layer_cls(self.d_model, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, received {feature_dim}")

        if seq_len == 0:
            return torch.zeros(
                batch,
                0,
                self.d_model,
                dtype=x.dtype,
                device=x.device,
            )

        self.reset_state()

        head_inputs = x.view(batch, seq_len, self.num_heads, self.head_dim)
        head_inputs = head_inputs.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        head_outputs = []
        for idx, head in enumerate(self.heads):
            head_input = head_inputs[:, idx, :, :]
            head_output = head(head_input)
            head_outputs.append(head_output)

        concatenated = torch.cat(head_outputs, dim=-1)
        return self.output(concatenated)

    def reset_state(self) -> None:
        for head in self.heads:
            cast(SSEAttention, head).reset_state()


class _MultiHeadStateAdapter:
    """Aggregates per-head SSE states into a single tensor view."""

    def __init__(self, heads: nn.ModuleList) -> None:
        self._heads = heads

    @property
    def states(self) -> Tensor:
        return torch.stack(
            [cast(SSEAttention, head).state_mgr.states for head in self._heads],
            dim=0,
        )

    def reset_state(self) -> None:
        for head in self._heads:
            cast(SSEAttention, head).reset_state()


class SSEMultiHeadAttention(nn.Module):
    """Batched SSE multi-head attention (optimized variant implemented later)."""

    def __init__(self, config: SSEMultiHeadAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self._naive = NaiveSSEMultiHeadAttention(config)
        self.state_mgr = _MultiHeadStateAdapter(self._naive.heads)
        self._query_modules = nn.ModuleList(
            [cast(SSEAttention, head).query for head in self._naive.heads]
        )
        self._key_modules = nn.ModuleList(
            [cast(SSEAttention, head).key for head in self._naive.heads]
        )
        self._value_modules = nn.ModuleList(
            [cast(SSEAttention, head).value for head in self._naive.heads]
        )

    @property
    def d_model(self) -> int:
        return self._naive.d_model

    @property
    def num_heads(self) -> int:
        return self._naive.num_heads

    @property
    def head_dim(self) -> int:
        return self._naive.head_dim

    @property
    def query(self) -> nn.ModuleList:
        return self._query_modules

    @property
    def key(self) -> nn.ModuleList:
        return self._key_modules

    @property
    def value(self) -> nn.ModuleList:
        return self._value_modules

    @property
    def output(self) -> nn.Module:
        return self._naive.output

    def forward(self, x: Tensor) -> Tensor:
        return self._naive(x)

    def reset_state(self) -> None:
        self._naive.reset_state()

    def load_state_from_naive(self, naive: NaiveSSEMultiHeadAttention) -> None:
        if naive.config != self.config:
            raise ValueError("Provided naive module configuration does not match.")
        self._naive.load_state_dict(naive.state_dict())


__all__ = [
    "NaiveSSEMultiHeadAttention",
    "SSEMultiHeadAttention",
    "SSEMultiHeadAttentionConfig",
]
