# Changelog

## v0.2.3 - 2026-03-24

### Fixed

- Python requirement lowered to `>=3.8` — enables use on Graphcore IPU nodes (Ubuntu 20.04, Python 3.8)

## v0.2.2 - 2026-03-24

### Fixed

- Torch requirement lowered to `>=2.0.0` — enables use with Graphcore PopTorch 3.3 and other constrained environments
- Output parity tests for `ButterflyMultiHeadAttention` now use `to_linear` roundtrip instead of `from_linear` fitting (which can't converge for arbitrary dense matrices)
- All pyrefly type errors resolved — `int()` wrapping for tensor `.item()` calls, typed casts for `nn.ModuleList` iteration
- `from_linear` docstring documents the convergence limitation

### Added

- Pre-commit hooks: ruff (lint + format) and pyrefly (type checking)
- GitHub Actions CI: test (Python 3.11 + 3.12), lint, typecheck — all blocking
- `py.typed` marker (PEP 561) for downstream type checking

## v0.2.1 - 2026-03-23

### Changed

- Package reorganized into `ops/`, `modules/`, `models/` following the Flash-Attention pattern
- Baseline implementations (`SimpleMLP`, `CustomMLP`, `CustomLinear`, `MultiHeadAttention`) moved to `models/_baselines.py` — still importable but clearly separated from the public API
- Config classes merged into their module files (no more separate `*_config.py` files)

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
