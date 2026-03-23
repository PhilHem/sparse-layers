# sparse-layers

Structured sparse layers for building memory-efficient neural networks on PyTorch. Drop-in replacements for standard layers using butterfly factorization, SSE attention, and other sparse primitives.

## Install

```bash
pip install sparse-layers
```

## Usage

```python
import torch
from sparse_layers import ButterflyLinear, ButterflyMLP

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
from sparse_layers import ButterflyMultiHeadAttention

# Multi-head attention with butterfly-factorized Q/K/V projections
bf_attn = ButterflyMultiHeadAttention(d_model=256, num_heads=8)

seq = torch.randn(32, 128, 256)  # (batch, seq_len, d_model)
out = bf_attn(seq, seq, seq)
```

### SSE Attention

State Space Exploration modules for efficient sequence modeling with sparse attention patterns.

```python
from sparse_layers.modules import SSEAttention, SSEAttentionConfig

config = SSEAttentionConfig(d_model=256, num_partitions=4)
sse = SSEAttention(config)

x = torch.randn(32, 128, 256)
out = sse(x)
```

## Architecture

Organized following the [Flash-Attention](https://github.com/Dao-AILab/flash-attention) pattern:

### Ops (`sparse_layers.ops`)

Primitive operations and utility functions.

| Module | Description |
|--------|-------------|
| `butterfly` | Butterfly factor multiply, power-of-2 utilities |
| `SSEMaskingOps` | Masking utilities for variable-length SSE |
| `SSEVarlenOps` | Variable-length sequence operations |

### Modules (`sparse_layers.modules`)

Composable building blocks — single units of computation.

| Module | Description |
|--------|-------------|
| `ButterflyLinear` | Linear layer using butterfly matrix factorization — O(n log n) parameters instead of O(n²) |
| `PaddedButterflyLinear` | ButterflyLinear with automatic padding for non-power-of-2 dimensions |
| `SSEAttention` | Sparse attention with state-space-inspired partitioning |
| `SSEAttentionAdaptive` | SSE with adaptive implementation selection (naive/batched) |
| `SSEMultiPartitionState` | Manages partition states across sequence chunks |
| `SSEPartitionSelector` | Selects active partitions per query position |
| `SSESparseSoftmax` | Sparse softmax over selected partitions |
| `LinearAttention` | Linear attention baseline (O(n) complexity) |

### Models (`sparse_layers.models`)

Composed architectures built from modules.

| Module | Description |
|--------|-------------|
| `ButterflyMLP` | Two-layer MLP with butterfly-factorized linear layers |
| `ButterflyMultiHeadAttention` | Multi-head attention with butterfly-factorized Q/K/V projections |
| `SSEMultiHeadAttention` | Multi-head variant of SSE attention |

The package also includes baseline implementations (`SimpleMLP`, `CustomMLP`, `CustomLinear`, `MultiHeadAttention`) used internally for validation and testing.

## License

MIT
