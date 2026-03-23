from __future__ import annotations

from torch import Tensor, nn

from sparse_layers.sse.partition_selector_config import SSEPartitionSelectorConfig


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

