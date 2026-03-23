from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SSEMultiPartitionStateConfig(BaseModel):
    """Configuration for the SSE multi-partition state manager."""

    model_config = ConfigDict(frozen=True)

    num_partitions: int = Field(
        gt=1, description="Number of state partitions"
    )
    c: int = Field(gt=0, description="State rows per partition")
    d: int = Field(gt=0, description="State feature dimension")
