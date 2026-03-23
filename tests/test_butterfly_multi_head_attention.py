import pytest
import torch

from sparse_layers import ButterflyLinear, ButterflyMultiHeadAttention, MultiHeadAttention


def test_forward_preserves_shape() -> None:
    module = ButterflyMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
    inputs = torch.randn(2, 5, 16)

    outputs = module(inputs)

    assert outputs.shape == (2, 5, 16)


def test_forward_supports_mask() -> None:
    module = ButterflyMultiHeadAttention(d_model=16, num_heads=4)
    inputs = torch.randn(3, 7, 16)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[:, -2:] = True

    outputs = module(inputs, mask=mask)

    assert outputs.shape == inputs.shape


def test_uses_butterfly_linear_layers() -> None:
    module = ButterflyMultiHeadAttention(d_model=16, num_heads=4)

    assert isinstance(module.query, ButterflyLinear)
    assert isinstance(module.key, ButterflyLinear)
    assert isinstance(module.value, ButterflyLinear)
    assert isinstance(module.out, ButterflyLinear)


def test_parameter_count_reduced_vs_baseline() -> None:
    baseline = MultiHeadAttention(d_model=16, num_heads=4)
    butterfly = ButterflyMultiHeadAttention(d_model=16, num_heads=4)

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    butterfly_params = sum(parameter.numel() for parameter in butterfly.parameters())

    assert butterfly_params < baseline_params


@pytest.mark.parametrize(
    "d_model, num_heads",
    [
        (0, 4),
        (16, 0),
        (18, 4),
    ],
)
def test_invalid_configurations_raise_value_error(d_model: int, num_heads: int) -> None:
    with pytest.raises(ValueError):
        ButterflyMultiHeadAttention(d_model=d_model, num_heads=num_heads)


def test_requires_power_of_two_d_model() -> None:
    with pytest.raises(ValueError, match="power of two"):
        ButterflyMultiHeadAttention(d_model=12, num_heads=4)


def test_to_linear_roundtrip_parity() -> None:
    """ButterflyMultiHeadAttention output should match its dense equivalent via to_linear."""
    torch.manual_seed(42)
    d_model, num_heads = 16, 4

    butterfly = ButterflyMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    butterfly.eval()

    # Build a dense baseline from the butterfly's own weights
    dense = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    dense.eval()

    with torch.no_grad():
        dense.query = butterfly.query.to_linear()
        dense.key = butterfly.key.to_linear()
        dense.value = butterfly.value.to_linear()
        dense.out = butterfly.out.to_linear()

    x = torch.randn(4, 8, d_model)

    with torch.no_grad():
        butterfly_out = butterfly(x)
        dense_out = dense(x)

    assert torch.allclose(butterfly_out, dense_out, atol=1e-4, rtol=1e-4), (
        f"Max diff: {(butterfly_out - dense_out).abs().max().item():.6f}"
    )


def test_to_linear_roundtrip_parity_with_mask() -> None:
    """Roundtrip parity should hold with masking applied."""
    torch.manual_seed(123)
    d_model, num_heads = 16, 4

    butterfly = ButterflyMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    butterfly.eval()

    dense = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    dense.eval()

    with torch.no_grad():
        dense.query = butterfly.query.to_linear()
        dense.key = butterfly.key.to_linear()
        dense.value = butterfly.value.to_linear()
        dense.out = butterfly.out.to_linear()

    x = torch.randn(4, 8, d_model)
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[:, -2:] = True

    with torch.no_grad():
        butterfly_out = butterfly(x, mask=mask)
        dense_out = dense(x, mask=mask)

    assert torch.allclose(butterfly_out, dense_out, atol=1e-4, rtol=1e-4), (
        f"Max diff: {(butterfly_out - dense_out).abs().max().item():.6f}"
    )
