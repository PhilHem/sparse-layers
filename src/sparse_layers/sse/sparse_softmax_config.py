from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SSESparseSoftmaxConfig(BaseModel):
    """Configuration for the SSE sparse softmax module."""

    model_config = ConfigDict(frozen=True)

    num_partitions: int = Field(
        gt=1, description="Total number of partitions in the SSE state matrix"
    )
    state_rows_per_partition: int = Field(
        gt=0,
        description="Number of state rows managed within each partition",
    )
    k: int = Field(
        gt=0,
        description="Number of partitions selected per token for sparse softmax",
    )

    @field_validator("k")
    @classmethod
    def validate_k_within_partitions(
        cls, value: int, info: ValidationInfo
    ) -> int:
        num_partitions = info.data.get("num_partitions")
        if num_partitions is not None and value > num_partitions:
            raise ValueError(
                f"k={value} must be <= num_partitions={num_partitions}"
            )
        return value




