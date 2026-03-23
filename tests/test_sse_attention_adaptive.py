import pytest
import torch
from pydantic import ValidationError

from butterfly_layers.sse import (
    SSEAttentionAdaptive,
    SSEAttentionAdaptiveConfig,
)


def _build_config(**overrides) -> SSEAttentionAdaptiveConfig:
    defaults = dict(
        d_model=32,
        num_partitions=6,
        k=3,
        state_rows=4,
        threshold=8,
        return_inverse=False,
    )
    defaults.update(overrides)
    return SSEAttentionAdaptiveConfig(**defaults)


def _build_module(**config_overrides) -> SSEAttentionAdaptive:
    config = _build_config(**config_overrides)
    return SSEAttentionAdaptive(config)


def test_adaptive_config_accepts_valid_values():
    config = _build_config()

    assert config.d_model == 32
    assert config.num_partitions == 6
    assert config.k == 3
    assert config.threshold == 8
    assert config.return_inverse is False


@pytest.mark.parametrize("threshold", [0, -1])
def test_adaptive_config_rejects_invalid_threshold(threshold):
    with pytest.raises(ValidationError):
        _build_config(threshold=threshold)


def test_adaptive_config_requires_k_within_partitions():
    with pytest.raises(ValidationError):
        _build_config(num_partitions=4, k=5)


def test_adaptive_config_is_immutable():
    config = _build_config()

    with pytest.raises(ValidationError):
        config.threshold = 16


def test_adaptive_uses_masking_below_threshold():
    torch.manual_seed(0)
    module = _build_module(threshold=16)
    x = torch.randn(2, 6, module.config.d_model)
    partition_indices = torch.randint(
        0, module.config.num_partitions, (2, 6, module.config.k)
    )

    output = module(x, partition_indices)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        2,
        6,
        module.config.num_partitions,
        module.config.d_model,
    )


@pytest.mark.parametrize("seq_len", [8, 12])
def test_adaptive_uses_varlen_at_or_above_threshold(seq_len):
    torch.manual_seed(1)
    module = _build_module(threshold=8, return_inverse=True)
    x = torch.randn(2, seq_len, module.config.d_model)
    partition_indices = torch.randint(
        0, module.config.num_partitions, (2, seq_len, module.config.k)
    )

    packed, cu_seqlens, inverse = module(x, partition_indices)

    assert isinstance(packed, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(inverse, torch.Tensor)
    assert cu_seqlens.shape == (
        module.config.num_partitions * x.size(0) + 1,
    )
    assert cu_seqlens.dtype == torch.int32
    assert inverse.shape == (2, seq_len, module.config.k)


def test_adaptive_handles_zero_length_sequence():
    module = _build_module(threshold=4)
    x = torch.randn(3, 0, module.config.d_model)
    partition_indices = torch.empty(3, 0, module.config.k, dtype=torch.long)

    output = module(x, partition_indices)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        3,
        0,
        module.config.num_partitions,
        module.config.d_model,
    )


def test_adaptive_forwards_validation_errors():
    module = _build_module()
    x = torch.randn(1, 2, module.config.d_model)
    partition_indices = torch.randn(1, 2, module.config.k)

    with pytest.raises(ValueError, match="partition_indices must have dtype"):
        module(x, partition_indices)


def test_adaptive_respects_return_inverse_flag():
    torch.manual_seed(2)
    module = _build_module(threshold=4, return_inverse=False)
    seq_len = 6
    x = torch.randn(1, seq_len, module.config.d_model)
    partition_indices = torch.randint(
        0, module.config.num_partitions, (1, seq_len, module.config.k)
    )

    _, _, inverse = module(x, partition_indices)

    assert inverse is None



