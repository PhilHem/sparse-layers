from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


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

