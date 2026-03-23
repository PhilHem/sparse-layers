import pytest
import torch
from pydantic import ValidationError

from sparse_layers.models import (
    NaiveSSEMultiHeadAttention,
    SSEMultiHeadAttention,
    SSEMultiHeadAttentionConfig,
)


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _build_config(**overrides) -> SSEMultiHeadAttentionConfig:
    defaults = dict(
        d_model=32,
        num_heads=4,
        num_partitions=6,
        k=3,
        state_rows=8,
        use_butterfly=False,
    )
    defaults.update(overrides)
    return SSEMultiHeadAttentionConfig(**defaults)


def test_multi_head_config_accepts_valid_values():
    config = _build_config()

    assert config.d_model == 32
    assert config.num_heads == 4
    assert config.num_partitions == 6
    assert config.k == 3
    assert config.state_rows == 8
    assert config.use_butterfly is False


@pytest.mark.parametrize("d_model", [0, -4])
def test_multi_head_config_requires_positive_d_model(d_model: int):
    with pytest.raises(ValidationError):
        _build_config(d_model=d_model)


@pytest.mark.parametrize("num_heads", [0, -2])
def test_multi_head_config_requires_positive_num_heads(num_heads: int):
    with pytest.raises(ValidationError):
        _build_config(num_heads=num_heads)


def test_multi_head_config_requires_divisible_d_model():
    with pytest.raises(ValidationError):
        _build_config(d_model=30, num_heads=4)


@pytest.mark.parametrize("num_partitions", [0, 1, -3])
def test_multi_head_config_requires_num_partitions_greater_than_one(num_partitions: int):
    with pytest.raises(ValidationError):
        _build_config(num_partitions=num_partitions)


@pytest.mark.parametrize("state_rows", [0, -5])
def test_multi_head_config_requires_positive_state_rows(state_rows: int):
    with pytest.raises(ValidationError):
        _build_config(state_rows=state_rows)


@pytest.mark.parametrize("k", [0, -1])
def test_multi_head_config_requires_positive_k(k: int):
    with pytest.raises(ValidationError):
        _build_config(k=k)


def test_multi_head_config_requires_k_less_or_equal_num_partitions():
    with pytest.raises(ValidationError):
        _build_config(num_partitions=4, k=5)


def test_multi_head_config_is_immutable():
    config = _build_config()

    with pytest.raises(ValidationError):
        config.k = 1


def _build_naive(**overrides) -> NaiveSSEMultiHeadAttention:
    config = _build_config(**overrides)
    return NaiveSSEMultiHeadAttention(config)


def _build_batched(**overrides) -> SSEMultiHeadAttention:
    config = _build_config(**overrides)
    return SSEMultiHeadAttention(config)


def test_multi_head_state_manager_shape_matches_config():
    attention = _build_batched(
        d_model=12,
        num_heads=3,
        num_partitions=5,
        k=2,
        state_rows=4,
    )

    expected_shape = (
        attention.num_heads,
        attention.config.num_partitions,
        attention.config.state_rows,
        attention.head_dim,
    )

    assert attention.state_mgr.states.shape == expected_shape


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_naive_forward_preserves_shape(num_heads: int):
    torch.manual_seed(0)
    d_model = 16
    batch, seq_len = 3, 5
    attention = _build_naive(d_model=d_model, num_heads=num_heads)
    inputs = torch.randn(batch, seq_len, attention.config.d_model)

    outputs = attention(inputs)

    assert outputs.shape == (batch, seq_len, attention.config.d_model)


def test_naive_state_reset_prevents_state_leakage():
    torch.manual_seed(1)
    attention = _build_naive(d_model=12, num_heads=3, num_partitions=5, k=2, state_rows=4)
    inputs = torch.randn(2, 4, attention.config.d_model)

    first = attention(inputs)
    second = attention(inputs)

    assert torch.allclose(first, second)


def test_naive_supports_float64_inputs():
    torch.manual_seed(2)
    attention = _build_naive(d_model=10, num_heads=2, num_partitions=4, k=2, state_rows=6)
    attention = attention.double()
    inputs = torch.randn(2, 3, attention.config.d_model, dtype=torch.float64)

    outputs = attention(inputs)

    assert outputs.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_naive_runs_on_cuda():
    torch.manual_seed(3)
    attention = _build_naive(d_model=8, num_heads=2, num_partitions=4, k=2, state_rows=4).cuda()
    inputs = torch.randn(2, 3, attention.config.d_model, device="cuda")

    outputs = attention(inputs)

    assert outputs.is_cuda


def test_naive_requires_three_dimensional_inputs():
    attention = _build_naive()
    inputs = torch.randn(attention.config.d_model)

    with pytest.raises(ValueError, match="expected input of shape"):
        attention(inputs)


def test_naive_requires_matching_feature_dimension():
    attention = _build_naive(d_model=18, num_heads=3)
    inputs = torch.randn(2, 4, attention.config.d_model + 1)

    with pytest.raises(ValueError, match="expected last dimension"):
        attention(inputs)


# -----------------------------------------------------------------------------
# Batched multi-head tests (RED phase)
# -----------------------------------------------------------------------------


def test_batched_forward_preserves_shape():
    attention = _build_batched(d_model=24, num_heads=3, num_partitions=5, k=2, state_rows=6)
    inputs = torch.randn(2, 4, attention.config.d_model)

    outputs = attention(inputs)

    assert outputs.shape == (2, 4, attention.config.d_model)


def test_batched_forward_matches_naive_outputs():
    torch.manual_seed(4)
    config_kwargs = dict(
        d_model=16,
        num_heads=4,
        num_partitions=5,
        k=2,
        state_rows=6,
    )
    naive = _build_naive(**config_kwargs)
    batched = _build_batched(**config_kwargs)
    batched.load_state_from_naive(naive)

    inputs = torch.randn(3, 5, config_kwargs["d_model"])

    expected = naive(inputs)
    actual = batched(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_batched_requires_three_dimensional_inputs():
    attention = _build_batched()
    inputs = torch.randn(attention.config.d_model)

    with pytest.raises(ValueError, match="expected input of shape"):
        attention(inputs)


def test_batched_requires_matching_feature_dimension():
    attention = _build_batched(d_model=18, num_heads=3)
    inputs = torch.randn(2, 4, attention.config.d_model + 2)

    with pytest.raises(ValueError, match="expected last dimension"):
        attention(inputs)


def test_batched_backward_propagates_gradients():
    torch.manual_seed(5)
    attention = _build_batched(d_model=20, num_heads=4, num_partitions=6, k=3, state_rows=5)
    inputs = torch.randn(2, 6, attention.config.d_model, requires_grad=True)

    outputs = attention(inputs)
    loss = outputs.pow(2).mean()
    loss.backward()

    tracked_modules = (
        attention.query,
        attention.key,
        attention.value,
        attention.output,
    )

    for module in tracked_modules:
        for param in module.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    assert torch.isfinite(inputs.grad).all()


def test_batched_handles_zero_sequence_length():
    attention = _build_batched(d_model=12, num_heads=3)
    inputs = torch.zeros(2, 0, attention.config.d_model)

    outputs = attention(inputs)

    assert outputs.shape == (2, 0, attention.config.d_model)


def test_batched_state_reset_clears_memory():
    torch.manual_seed(6)
    attention = _build_batched(d_model=12, num_heads=3, num_partitions=5, k=2, state_rows=4)
    inputs = torch.randn(2, 4, attention.config.d_model)

    attention(inputs)
    assert torch.any(attention.state_mgr.states != 0)

    attention.reset_state()

    assert torch.allclose(
        attention.state_mgr.states,
        torch.zeros_like(attention.state_mgr.states),
    )


def test_batched_supports_float64_inputs():
    torch.manual_seed(7)
    attention = _build_batched(d_model=10, num_heads=2, num_partitions=4, k=2, state_rows=5).double()
    inputs = torch.randn(2, 3, attention.config.d_model, dtype=torch.float64)

    outputs = attention(inputs)

    assert outputs.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_runs_on_cuda():
    torch.manual_seed(8)
    attention = _build_batched(d_model=16, num_heads=4, num_partitions=6, k=3, state_rows=6).cuda()
    inputs = torch.randn(2, 3, attention.config.d_model, device="cuda")

    outputs = attention(inputs)

    assert outputs.is_cuda


def test_batched_with_butterfly_reduces_parameter_count():
    kwargs = dict(d_model=32, num_heads=4, num_partitions=6, k=2, state_rows=8)
    dense = _build_batched(**kwargs, use_butterfly=False)
    butterfly = _build_batched(**kwargs, use_butterfly=True)

    assert _count_trainable_parameters(butterfly) < _count_trainable_parameters(dense)


