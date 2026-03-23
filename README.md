# butterfly-layers

Butterfly-factorized neural network layers for memory-efficient deep learning on PyTorch.

## Install

```bash
pip install butterfly-layers
```

## Usage

```python
import torch
from butterfly_layers import ButterflyLinear, ButterflyMLP

# Drop-in replacement for nn.Linear with O(n log n) parameters
layer = ButterflyLinear(in_features=256, out_features=256)
x = torch.randn(32, 256)
y = layer(x)

# MLP with butterfly-factorized linear layers
mlp = ButterflyMLP(in_features=256, hidden_features=512, out_features=256)
y = mlp(x)
```

### Attention

```python
from butterfly_layers import ButterflyMultiHeadAttention, MultiHeadAttention

# Standard multi-head attention
attn = MultiHeadAttention(d_model=256, num_heads=8)

# Butterfly-factorized variant (fewer parameters, same interface)
bf_attn = ButterflyMultiHeadAttention(d_model=256, num_heads=8)

seq = torch.randn(32, 128, 256)  # (batch, seq_len, d_model)
out = bf_attn(seq, seq, seq)
```

### SSE Attention

State Space Exploration modules for efficient sequence modeling with sparse attention patterns.

```python
from butterfly_layers.sse import SSEAttention, SSEAttentionConfig

config = SSEAttentionConfig(d_model=256, num_partitions=4)
sse = SSEAttention(config)

x = torch.randn(32, 128, 256)
out = sse(x)
```

## Modules

### Layers (`butterfly_layers.layers`)

| Module | Description |
|--------|-------------|
| `ButterflyLinear` | Linear layer using butterfly matrix factorization — O(n log n) parameters instead of O(n²) |
| `PaddedButterflyLinear` | ButterflyLinear with automatic padding for non-power-of-2 dimensions |
| `ButterflyMLP` | Two-layer MLP with butterfly-factorized linear layers |
| `ButterflyMultiHeadAttention` | Multi-head attention with butterfly-factorized Q/K/V projections |
| `MultiHeadAttention` | Standard multi-head attention (baseline) |
| `CustomLinear` | Linear layer with pluggable weight initialization |
| `CustomMLP` | MLP with CustomLinear layers |
| `SimpleMLP` | Minimal MLP baseline |

### SSE (`butterfly_layers.sse`)

| Module | Description |
|--------|-------------|
| `SSEAttention` | Sparse attention with state-space-inspired partitioning |
| `SSEAttentionAdaptive` | SSE with adaptive implementation selection (naive/batched) |
| `SSEMultiHeadAttention` | Multi-head variant of SSE attention |
| `SSEMultiPartitionState` | Manages partition states across sequence chunks |
| `SSEPartitionSelector` | Selects active partitions per query position |
| `SSESparseSoftmax` | Sparse softmax over selected partitions |
| `LinearAttention` | Linear attention baseline (O(n) complexity) |
| `SSEMaskingOps` | Masking utilities for variable-length SSE |
| `SSEVarlenOps` | Variable-length sequence operations |

## License

MIT
