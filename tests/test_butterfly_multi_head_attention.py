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


def test_output_parity_with_baseline_via_from_linear() -> None:
    """ButterflyMultiHeadAttention fitted from a baseline should produce equivalent output."""
    torch.manual_seed(42)
    d_model, num_heads = 16, 4

    baseline = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    baseline.eval()

    butterfly = ButterflyMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    butterfly.eval()

    # Fit each butterfly projection from the corresponding baseline projection
    # (from_linear uses gradient-based optimization internally — no torch.no_grad())
    butterfly.query = ButterflyLinear.from_linear(baseline.query, seed=0)
    butterfly.key = ButterflyLinear.from_linear(baseline.key, seed=1)
    butterfly.value = ButterflyLinear.from_linear(baseline.value, seed=2)
    butterfly.out = ButterflyLinear.from_linear(baseline.out, seed=3)

    x = torch.randn(4, 8, d_model)

    with torch.no_grad():
        baseline_out = baseline(x)
        butterfly_out = butterfly(x)

    assert torch.allclose(baseline_out, butterfly_out, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(baseline_out - butterfly_out).abs().max().item():.6f}"
    )


def test_output_parity_with_mask() -> None:
    """Parity should hold with masking applied."""
    torch.manual_seed(123)
    d_model, num_heads = 16, 4

    baseline = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    baseline.eval()

    butterfly = ButterflyMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    butterfly.eval()

    butterfly.query = ButterflyLinear.from_linear(baseline.query, seed=0)
    butterfly.key = ButterflyLinear.from_linear(baseline.key, seed=1)
    butterfly.value = ButterflyLinear.from_linear(baseline.value, seed=2)
    butterfly.out = ButterflyLinear.from_linear(baseline.out, seed=3)

    x = torch.randn(4, 8, d_model)
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[:, -2:] = True

    with torch.no_grad():
        baseline_out = baseline(x, mask=mask)
        butterfly_out = butterfly(x, mask=mask)

    assert torch.allclose(baseline_out, butterfly_out, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(baseline_out - butterfly_out).abs().max().item():.6f}"
    )

