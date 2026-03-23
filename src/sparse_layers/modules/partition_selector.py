from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from torch import Tensor, nn


class SSEPartitionSelectorConfig(BaseModel):
    """Configuration for the SSE partition selection module."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Model dimension of the input features")
    num_partitions: int = Field(
        gt=1, description="Total number of memory partitions available"
    )
    k: int = Field(gt=0, description="Number of partitions to select per token")

    @field_validator("k")
    @classmethod
    def validate_k_within_partitions(cls, value: int, info: ValidationInfo) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(
                f"k={value} must be <= num_partitions={num_partitions}"
            )
        return value


class SSEPartitionSelector(nn.Module):
    """Naive top-k partition selection module for SSE attention."""

    def __init__(self, config: SSEPartitionSelectorConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.k = config.k
        self.num_partitions = config.num_partitions

        self.linear = nn.Linear(self.d_model, self.num_partitions)

    def forward(self, x: Tensor) -> Tensor:
        """Select the top-k partitions for each token in the sequence."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape (batch, seq_len, d_model), "
                f"received tensor with {x.ndim} dimensions"
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected final dimension {self.d_model}, found {x.shape[-1]}"
            )

        bias_scores = self.linear(x)
        _, top_indices = bias_scores.topk(self.k, dim=-1)
        return top_indices

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_partitions={self.num_partitions}, k={self.k}"
        )
