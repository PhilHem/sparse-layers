import pytest
import torch


def _build_module(**kwargs):
    from sparse_layers import MultiHeadAttention

    return MultiHeadAttention(**kwargs)


def test_invalid_head_configuration_raises_value_error():
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        _build_module(d_model=15, num_heads=4)


@pytest.mark.parametrize("batch_size, seq_len", [(1, 4), (3, 7)])
def test_forward_preserves_input_shape(batch_size, seq_len):
    module = _build_module(d_model=16, num_heads=4, dropout=0.0)
    inputs = torch.randn(batch_size, seq_len, 16)

    outputs = module(inputs)

    assert outputs.shape == (batch_size, seq_len, 16)


def test_mask_shape_mismatch_raises_value_error():
    module = _build_module(d_model=8, num_heads=2)
    inputs = torch.randn(2, 5, 8)
    mask = torch.zeros(3, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape must match"):
        module(inputs, mask=mask)

