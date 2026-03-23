from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LinearAttentionConfig(BaseModel):
    """Configuration for the baseline linear attention module."""

    model_config = ConfigDict(frozen=True)

    d_model: int = Field(gt=0, description="Feature dimension of the input tensors")
    d_state: int = Field(gt=0, description="Number of rows in the compressed state")
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied to attention outputs",
    )




