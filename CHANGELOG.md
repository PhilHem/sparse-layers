# Changelog

## v0.2.0 - 2026-03-23

Initial public release as `sparse-layers` (renamed from `butterfly-layers`).

### Added

- Butterfly-factorized linear layer (`ButterflyLinear`) — O(n log n) parameters as a drop-in replacement for `nn.Linear`
- `PaddedButterflyLinear` for non-power-of-2 dimensions
- `ButterflyMLP` and `ButterflyMultiHeadAttention` — sparse MLP and attention building blocks
- SSE (State Space Exploration) attention modules: `SSEAttention`, `SSEAttentionAdaptive`, `SSEMultiHeadAttention`
- SSE infrastructure: partition selector, sparse softmax, multi-partition state management
- `LinearAttention` baseline with O(n) complexity
- Variable-length sequence operations (`SSEVarlenOps`) and masking utilities (`SSEMaskingOps`)
- Pydantic-based configuration for all SSE modules
- Boundary enforcement test — verifies the package has no infrastructure dependencies
