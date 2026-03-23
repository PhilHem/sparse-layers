import pytest
import torch
from pydantic import ValidationError

from sparse_layers.sse import (
    NaiveSSEAttention,
    SSEAttention,
    SSEAttentionConfig,
)
from sparse_layers import PaddedButterflyLinear


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _build_config(**overrides):
    defaults = dict(
        d_model=32,
        num_partitions=4,
        k=2,
        state_rows=16,
        use_butterfly=False,
    )
    defaults.update(overrides)
    return SSEAttentionConfig(**defaults)


def test_sse_attention_config_accepts_valid_values():
    config = _build_config()

    assert config.d_model == 32
    assert config.num_partitions == 4
    assert config.k == 2
    assert config.state_rows == 16
    assert config.use_butterfly is False


@pytest.mark.parametrize("d_model", [0, -1])
def test_sse_attention_config_requires_positive_d_model(d_model):
    with pytest.raises(ValidationError):
        _build_config(d_model=d_model)


@pytest.mark.parametrize("num_partitions", [0, 1, -5])
def test_sse_attention_config_requires_num_partitions_greater_than_one(num_partitions):
    with pytest.raises(ValidationError):
        _build_config(num_partitions=num_partitions)


@pytest.mark.parametrize("state_rows", [0, -3])
def test_sse_attention_config_requires_positive_state_rows(state_rows):
    with pytest.raises(ValidationError):
        _build_config(state_rows=state_rows)


@pytest.mark.parametrize("k", [0, -2])
def test_sse_attention_config_requires_positive_k(k):
    with pytest.raises(ValidationError):
        _build_config(k=k)


def test_sse_attention_config_requires_k_less_or_equal_num_partitions():
    with pytest.raises(ValidationError):
        _build_config(num_partitions=4, k=5)


def test_sse_attention_config_model_dump_matches_values():
    config = _build_config(
        d_model=128,
        num_partitions=8,
        k=4,
        state_rows=32,
        use_butterfly=True,
    )

    dumped = config.model_dump()

    assert dumped["d_model"] == 128
    assert dumped["num_partitions"] == 8
    assert dumped["k"] == 4
    assert dumped["state_rows"] == 32
    assert dumped["use_butterfly"] is True


def test_sse_attention_config_is_immutable():
    config = _build_config()

    with pytest.raises(ValidationError):
        config.k = 1


# -----------------------------------------------------------------------------
# Naive SSE attention tests (RED phase)
# -----------------------------------------------------------------------------


def _build_naive_attention(**overrides) -> NaiveSSEAttention:
    config = _build_config(**overrides)
    return NaiveSSEAttention(config)


def _build_batched_attention(**overrides) -> SSEAttention:
    config = _build_config(**overrides)
    return SSEAttention(config)


def test_naive_sse_attention_initializes_state_manager_shape():
    attention = _build_naive_attention(
        d_model=12,
        num_partitions=5,
        state_rows=6,
        k=3,
    )

    expected_shape = (
        attention.config.num_partitions,
        attention.config.state_rows,
        attention.config.d_model,
    )

    assert attention.state_mgr.states.shape == expected_shape


def test_sse_attention_initializes_state_manager_shape():
    attention = _build_batched_attention(
        d_model=10,
        num_partitions=4,
        state_rows=7,
        k=2,
    )

    expected_shape = (
        attention.config.num_partitions,
        attention.config.state_rows,
        attention.config.d_model,
    )

    assert attention.state_mgr.states.shape == expected_shape


def test_naive_sse_attention_forward_matches_manual_loop():
    torch.manual_seed(0)
    batch, seq_len, d_model = 2, 4, 8
    state_rows = 4
    attention = _build_naive_attention(
        d_model=d_model,
        state_rows=state_rows,
        num_partitions=5,
        k=2,
    )
    inputs = torch.randn(batch, seq_len, d_model)

    with torch.no_grad():
        # Manual baseline using same submodules
        attention.state_mgr.reset_state()
        manual_outputs = []
        for t in range(seq_len):
            token = inputs[:, t, :]
            token_seq = token.unsqueeze(1)
            partition_indices = attention.partition_selector(token_seq)

            q_t = attention.query(token)
            k_t = attention.key(token)
            v_t = attention.value(token)

            k_sparse = attention.sparse_softmax(k_t.unsqueeze(1), partition_indices)
            gather_index = partition_indices.unsqueeze(-1).expand(
                -1, -1, -1, state_rows
            )
            selected = torch.gather(k_sparse, dim=2, index=gather_index)

            attention.state_mgr.update(
                partition_indices,
                selected,
                v_t.unsqueeze(1),
            )

            read = attention.state_mgr.read(
                partition_indices,
                q_t.unsqueeze(1),
            )
            manual_outputs.append(read.squeeze(1))

        manual = torch.stack(manual_outputs, dim=1)
        manual = attention.output(manual)

    actual = attention(inputs)

    assert torch.allclose(actual, manual)


def test_naive_sse_attention_requires_three_dimensional_inputs():
    attention = _build_naive_attention()
    inputs = torch.randn(attention.config.d_model)

    with pytest.raises(ValueError, match="expected input of shape"):
        attention(inputs)


def test_naive_sse_attention_requires_matching_feature_dimension():
    attention = _build_naive_attention(d_model=6)
    inputs = torch.randn(2, 3, attention.config.d_model + 1)

    with pytest.raises(ValueError, match="expected last dimension"):
        attention(inputs)


def test_naive_sse_attention_backward_propagates_gradients():
    torch.manual_seed(1)
    attention = _build_naive_attention(d_model=6, state_rows=4, num_partitions=4, k=2)
    inputs = torch.randn(3, 5, attention.config.d_model, requires_grad=True)

    output = attention(inputs)
    loss = output.pow(2).mean()
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


def test_naive_sse_attention_supports_float64_inputs():
    torch.manual_seed(2)
    attention = _build_naive_attention(d_model=5, state_rows=4, num_partitions=4, k=2)
    attention = attention.double()
    inputs = torch.randn(2, 3, attention.config.d_model, dtype=torch.float64)

    outputs = attention(inputs)

    assert outputs.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_naive_sse_attention_runs_on_cuda():
    torch.manual_seed(3)
    attention = _build_naive_attention(d_model=6, state_rows=4, num_partitions=4, k=2).cuda()
    inputs = torch.randn(2, 4, attention.config.d_model, device="cuda")

    outputs = attention(inputs)

    assert outputs.is_cuda


def test_naive_sse_attention_state_does_not_leak_between_calls():
    torch.manual_seed(4)
    attention = _build_naive_attention(d_model=4, state_rows=3, num_partitions=4, k=2)
    inputs = torch.randn(1, 2, attention.config.d_model)

    first = attention(inputs)
    second = attention(inputs)

    assert torch.allclose(first, second)


# -----------------------------------------------------------------------------
# Batched SSE attention tests (RED phase)
# -----------------------------------------------------------------------------


def test_sse_attention_forward_preserves_shape():
    attention = _build_batched_attention(
        d_model=10,
        state_rows=6,
        num_partitions=5,
        k=3,
    )
    inputs = torch.randn(2, 4, attention.config.d_model)

    outputs = attention(inputs)

    assert outputs.shape == (2, 4, attention.config.d_model)


def test_sse_attention_matches_naive_outputs():
    torch.manual_seed(5)
    config_kwargs = dict(
        d_model=12,
        state_rows=6,
        num_partitions=5,
        k=3,
    )
    naive = _build_naive_attention(**config_kwargs)
    batched = _build_batched_attention(**config_kwargs)
    batched.load_state_dict(naive.state_dict())

    inputs = torch.randn(3, 7, config_kwargs["d_model"])

    expected = naive(inputs)
    actual = batched(inputs)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_sse_attention_requires_three_dimensional_inputs():
    attention = _build_batched_attention()
    inputs = torch.randn(attention.config.d_model)

    with pytest.raises(ValueError, match="expected input of shape"):
        attention(inputs)


def test_sse_attention_requires_matching_feature_dimension():
    attention = _build_batched_attention(d_model=9)
    inputs = torch.randn(2, 3, attention.config.d_model + 2)

    with pytest.raises(ValueError, match="expected last dimension"):
        attention(inputs)


def test_sse_attention_backward_propagates_gradients():
    torch.manual_seed(6)
    attention = _build_batched_attention(
        d_model=8,
        state_rows=4,
        num_partitions=4,
        k=2,
    )
    inputs = torch.randn(2, 5, attention.config.d_model, requires_grad=True)

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


def test_sse_attention_handles_zero_sequence_length():
    attention = _build_batched_attention(d_model=6)
    inputs = torch.zeros(3, 0, attention.config.d_model)

    outputs = attention(inputs)

    assert outputs.shape == (3, 0, attention.config.d_model)


def test_sse_attention_state_reset_clears_memory():
    torch.manual_seed(7)
    attention = _build_batched_attention(d_model=6, state_rows=4, num_partitions=4, k=2)
    inputs = torch.randn(2, 3, attention.config.d_model)

    attention(inputs)
    assert torch.any(attention.state_mgr.states != 0)

    attention.reset_state()

    assert torch.allclose(
        attention.state_mgr.states,
        torch.zeros_like(attention.state_mgr.states),
    )


def test_sse_attention_state_does_not_leak_between_calls():
    torch.manual_seed(8)
    attention = _build_batched_attention(d_model=6, state_rows=4, num_partitions=4, k=2)
    inputs = torch.randn(2, 4, attention.config.d_model)

    first = attention(inputs)
    second = attention(inputs)

    assert torch.allclose(first, second)


def test_sse_attention_supports_float64_inputs():
    torch.manual_seed(9)
    attention = _build_batched_attention(d_model=5, state_rows=4, num_partitions=4, k=2).double()
    inputs = torch.randn(2, 3, attention.config.d_model, dtype=torch.float64)

    outputs = attention(inputs)

    assert outputs.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sse_attention_runs_on_cuda():
    torch.manual_seed(10)
    attention = _build_batched_attention(d_model=6, state_rows=4, num_partitions=4, k=2).cuda()
    inputs = torch.randn(2, 3, attention.config.d_model, device="cuda")

    outputs = attention(inputs)

    assert outputs.is_cuda


def test_sse_attention_sparse_weights_are_normalized():
    torch.manual_seed(11)
    attention = _build_batched_attention(d_model=6, state_rows=4, num_partitions=5, k=3)
    inputs = torch.randn(2, 4, attention.config.d_model)

    with torch.no_grad():
        indices = attention.partition_selector(inputs)
        keys = attention.key(inputs)
        weights = attention.sparse_softmax(keys, indices)
        summed = weights.sum(dim=2)
        expected = torch.full_like(summed, 1.0 / attention.config.k)

    assert torch.allclose(summed, expected, atol=1e-4, rtol=1e-4)


def test_naive_sse_attention_with_butterfly_uses_padded_projections():
    attention = _build_naive_attention(use_butterfly=True)

    assert isinstance(attention.value, PaddedButterflyLinear)
    assert isinstance(attention.output, PaddedButterflyLinear)
    assert isinstance(attention.query, torch.nn.Linear)
    assert isinstance(attention.key, torch.nn.Linear)


def test_sse_attention_with_butterfly_uses_padded_projections():
    attention = _build_batched_attention(use_butterfly=True)

    assert isinstance(attention.value, PaddedButterflyLinear)
    assert isinstance(attention.output, PaddedButterflyLinear)
    assert isinstance(attention.query, torch.nn.Linear)
    assert isinstance(attention.key, torch.nn.Linear)


def test_naive_sse_attention_with_butterfly_reduces_parameter_count():
    config_kwargs = dict(d_model=32, state_rows=16, num_partitions=6, k=3)
    dense = _build_naive_attention(**config_kwargs, use_butterfly=False)
    butterfly = _build_naive_attention(**config_kwargs, use_butterfly=True)

    assert _count_trainable_parameters(butterfly) < _count_trainable_parameters(dense)


def test_sse_attention_with_butterfly_reduces_parameter_count():
    config_kwargs = dict(d_model=32, state_rows=16, num_partitions=6, k=3)
    dense = _build_batched_attention(**config_kwargs, use_butterfly=False)
    butterfly = _build_batched_attention(**config_kwargs, use_butterfly=True)

    assert _count_trainable_parameters(butterfly) < _count_trainable_parameters(dense)


def test_naive_sse_attention_butterfly_outputs_close_to_linear():
    torch.manual_seed(21)
    config_kwargs = dict(d_model=32, state_rows=16, num_partitions=5, k=2)
    dense = _build_naive_attention(**config_kwargs, use_butterfly=False)
    butterfly = _build_naive_attention(**config_kwargs, use_butterfly=True)

    inputs = torch.randn(2, 6, config_kwargs["d_model"])
    dense_out = dense(inputs)
    butterfly_out = butterfly(inputs)

    mse = torch.mean((dense_out - butterfly_out) ** 2)
    assert mse.item() < 10.0


def test_naive_sse_attention_butterfly_gradient_magnitudes_reasonable():
    torch.manual_seed(22)
    config_kwargs = dict(d_model=32, state_rows=16, num_partitions=5, k=2)
    dense = _build_naive_attention(**config_kwargs, use_butterfly=False)
    butterfly = _build_naive_attention(**config_kwargs, use_butterfly=True)

    dense_inputs = torch.randn(2, 6, config_kwargs["d_model"], requires_grad=True)
    butterfly_inputs = dense_inputs.clone().detach().requires_grad_(True)

    dense_loss = dense(dense_inputs).pow(2).mean()
    butterfly_loss = butterfly(butterfly_inputs).pow(2).mean()

    dense_loss.backward()
    butterfly_loss.backward()

    dense_grad_norm = torch.stack(
        [param.grad.norm() for param in dense.parameters() if param.grad is not None]
    ).mean()
    butterfly_grad_norm = torch.stack(
        [
            param.grad.norm()
            for param in butterfly.parameters()
            if param.grad is not None
        ]
    ).mean()

    ratio = torch.abs(dense_grad_norm - butterfly_grad_norm) / dense_grad_norm
    assert ratio.item() < 5.0
    assert torch.isfinite(butterfly_inputs.grad).all()
    assert torch.isfinite(dense_inputs.grad).all()


