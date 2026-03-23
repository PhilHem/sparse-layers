from __future__ import annotations

import torch
from torch import Tensor, nn

from sparse_layers.sse.linear_attention_config import LinearAttentionConfig


class LinearAttention(nn.Module):
    """Baseline linear attention module (forward implemented in later steps)."""

    def __init__(self, config: LinearAttentionConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state

        self.query = nn.Linear(self.d_model, self.d_state)
        self.key = nn.Linear(self.d_model, self.d_state)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.output = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, seq_len, d_model)")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, received {feature_dim}"
            )

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        state = torch.zeros(
            batch_size,
            self.d_state,
            self.d_model,
            dtype=value.dtype,
            device=value.device,
        )

        outputs = []
        for t in range(seq_len):
            k_t = key[:, t, :]
            v_t = value[:, t, :]
            update = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            state = state + update

            q_t = query[:, t, :]
            output_t = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)
            outputs.append(output_t)

        stacked = torch.stack(outputs, dim=1)
        stacked = self.dropout(stacked)
        return self.output(stacked)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"dropout={self.config.dropout}"
        )


