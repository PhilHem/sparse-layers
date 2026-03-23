import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from sparse_layers.modules import (
    SSEPartitionSelector,
    SSEPartitionSelectorConfig,
    SSESparseSoftmax,
    SSESparseSoftmaxConfig,
)


def test_config_accepts_valid_values():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=128, k=2
    )

    assert config.num_partitions == 4
    assert config.state_rows_per_partition == 128
    assert config.k == 2


@pytest.mark.parametrize("num_partitions", [0, 1, -3])
def test_config_requires_num_partitions_greater_than_one(num_partitions):
    with pytest.raises(ValidationError):
        SSESparseSoftmaxConfig(
            num_partitions=num_partitions, state_rows_per_partition=64, k=2
        )


@pytest.mark.parametrize("state_rows_per_partition", [0, -1, -16])
def test_config_requires_positive_state_rows(state_rows_per_partition):
    with pytest.raises(ValidationError):
        SSESparseSoftmaxConfig(
            num_partitions=4,
            state_rows_per_partition=state_rows_per_partition,
            k=2,
        )


@pytest.mark.parametrize("k", [0, -1, -7])
def test_config_requires_positive_k(k):
    with pytest.raises(ValidationError):
        SSESparseSoftmaxConfig(
            num_partitions=4,
            state_rows_per_partition=64,
            k=k,
        )


def test_config_requires_k_less_or_equal_num_partitions():
    with pytest.raises(ValidationError):
        SSESparseSoftmaxConfig(
            num_partitions=4, state_rows_per_partition=128, k=5
        )


def test_config_is_serializable():
    config = SSESparseSoftmaxConfig(
        num_partitions=6, state_rows_per_partition=32, k=3
    )

    dumped = config.model_dump()

    assert dumped == {
        "num_partitions": 6,
        "state_rows_per_partition": 32,
        "k": 3,
    }


def test_config_is_immutable():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=32, k=2
    )

    with pytest.raises(ValidationError):
        config.k = 1


def test_module_initializes_with_config():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )

    module = SSESparseSoftmax(config)

    assert module.config == config
    assert module.num_partitions == config.num_partitions
    assert module.state_rows_per_partition == config.state_rows_per_partition
    assert module.k == config.k


def test_module_exposes_no_parameters_by_default():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )

    module = SSESparseSoftmax(config)

    parameters = list(module.parameters())

    assert parameters == []


def test_module_repr_includes_key_attributes():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )

    module = SSESparseSoftmax(config)

    representation = repr(module)

    assert "num_partitions=4" in representation
    assert "state_rows_per_partition=64" in representation
    assert "k=2" in representation


def test_forward_requires_three_dimensional_keys():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 64)
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k))

    with pytest.raises(ValueError, match="Expected keys with shape"):
        module(keys, partition_indices)


def test_forward_requires_three_dimensional_partition_indices():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 32)
    partition_indices = torch.randint(0, config.num_partitions, (2, config.k))

    with pytest.raises(ValueError, match="Expected partition_indices with shape"):
        module(keys, partition_indices)


def test_forward_requires_matching_batch_dimension():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 32)
    partition_indices = torch.randint(0, config.num_partitions, (3, 3, config.k))

    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        module(keys, partition_indices)


def test_forward_requires_matching_sequence_length():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 32)
    partition_indices = torch.randint(0, config.num_partitions, (2, 4, config.k))

    with pytest.raises(ValueError, match="Sequence length mismatch"):
        module(keys, partition_indices)


def test_forward_requires_partition_k_matches_config():
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 32)
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k + 1))

    with pytest.raises(ValueError, match="Expected partition_indices last dimension"):
        module(keys, partition_indices)


def test_forward_returns_expected_shape():
    torch.manual_seed(0)
    batch, seq_len, d_model = 2, 5, 16
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    output = module(keys, partition_indices)

    assert output.shape == (batch, seq_len, config.num_partitions, d_model)
    assert output.dtype == keys.dtype


def test_forward_zeroes_out_non_selected_partitions():
    torch.manual_seed(1)
    batch, seq_len, d_model = 2, 4, 8
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    output = module(keys, partition_indices)

    all_indices = torch.arange(config.num_partitions)
    mask = torch.isin(all_indices.view(1, 1, -1), partition_indices)
    non_selected = (~mask).unsqueeze(-1)

    assert torch.allclose(
        output * non_selected, torch.zeros_like(output), atol=0.0, rtol=0.0
    )


def test_forward_is_deterministic_for_same_inputs():
    torch.manual_seed(2)
    batch, seq_len, d_model = 2, 3, 12
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    first = module(keys, partition_indices)
    second = module(keys, partition_indices)

    assert torch.allclose(first, second)


def test_forward_non_selected_partitions_remain_zero_across_calls():
    torch.manual_seed(14)
    batch, seq_len, d_model = 2, 3, 6
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    for _ in range(3):
        output = module(keys, partition_indices)
        all_indices = torch.arange(config.num_partitions)
        mask = torch.isin(all_indices.view(1, 1, -1), partition_indices)
        non_selected = (~mask).unsqueeze(-1)

        assert torch.allclose(
            output * non_selected, torch.zeros_like(output), atol=0.0, rtol=0.0
        )


def test_forward_handles_k_equals_one():
    torch.manual_seed(15)
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=1
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 5)
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k))

    output = module(keys, partition_indices)

    assert output.shape == (2, 3, config.num_partitions, 5)
    sums = output.sum(dim=2)
    expected = torch.full_like(sums, 1.0 / config.k)
    assert torch.allclose(sums, expected, atol=1e-4, rtol=1e-4)


def test_forward_handles_k_equals_num_partitions():
    torch.manual_seed(16)
    config = SSESparseSoftmaxConfig(
        num_partitions=3, state_rows_per_partition=64, k=3
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(1, 2, 4)
    partition_indices = torch.arange(config.num_partitions, dtype=torch.long)
    partition_indices = partition_indices.view(1, 1, -1).repeat(1, 2, 1)

    output = module(keys, partition_indices)
    reference = _reference_sparse_softmax(keys, partition_indices, config.num_partitions)

    assert torch.allclose(output, reference / config.k, atol=1e-6)


def test_forward_handles_single_batch_and_token():
    torch.manual_seed(17)
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(1, 1, 6)
    partition_indices = torch.randint(0, config.num_partitions, (1, 1, config.k))

    output = module(keys, partition_indices)

    assert output.shape == (1, 1, config.num_partitions, 6)
    assert torch.all(output >= 0)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_forward_probabilities_sum_to_expected_value(k):
    torch.manual_seed(3)
    batch, seq_len, d_model = 2, 3, 10
    num_partitions = max(4, k)
    config = SSESparseSoftmaxConfig(
        num_partitions=num_partitions, state_rows_per_partition=64, k=k
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, num_partitions, (batch, seq_len, k))

    output = module(keys, partition_indices)

    sums = output.sum(dim=2)
    expected = torch.full_like(sums, 1.0 / k)

    assert torch.allclose(sums, expected, atol=1e-4, rtol=1e-4)


def test_forward_probabilities_have_no_nans():
    torch.manual_seed(4)
    batch, seq_len, d_model = 2, 3, 10
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    output = module(keys, partition_indices)

    assert torch.all(torch.isfinite(output))
    assert torch.all(output >= 0)


def _reference_sparse_softmax(keys: torch.Tensor, partition_indices: torch.Tensor, num_partitions: int) -> torch.Tensor:
    expanded = keys.unsqueeze(2).expand(-1, -1, num_partitions, -1)
    partition_scale = torch.linspace(
        1.0,
        1.0 + (num_partitions - 1),
        steps=num_partitions,
        device=keys.device,
        dtype=keys.dtype,
    )
    scaled = expanded * partition_scale.view(1, 1, num_partitions, 1)
    full_softmax = F.softmax(scaled, dim=2)
    mask = keys.new_zeros(keys.shape[0], keys.shape[1], num_partitions)
    mask.scatter_(2, partition_indices, 1.0)
    masked = full_softmax * mask.unsqueeze(-1)
    normalization = masked.sum(dim=2, keepdim=True)
    eps = torch.finfo(masked.dtype).eps
    return masked / (normalization + eps)


def test_scaling_factor_matches_reference_normalized_values():
    torch.manual_seed(5)
    batch, seq_len, d_model = 2, 3, 10
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(batch, seq_len, d_model)
    partition_indices = torch.randint(0, config.num_partitions, (batch, seq_len, config.k))

    output = module(keys, partition_indices)
    reference = _reference_sparse_softmax(keys, partition_indices, config.num_partitions)

    assert torch.allclose(output, reference / config.k, atol=1e-6)


def test_gradient_magnitudes_reduce_with_larger_k():
    torch.manual_seed(6)
    batch, seq_len, d_model = 1, 2, 4
    base_keys = torch.randn(batch, seq_len, d_model)
    num_partitions = 5
    weight = torch.randn(batch, seq_len, num_partitions, d_model)

    k_values = (2, 3, 5)
    ratios = []

    for k in k_values:
        config = SSESparseSoftmaxConfig(
            num_partitions=num_partitions, state_rows_per_partition=64, k=k
        )
        module = SSESparseSoftmax(config)

        partition_indices = (
            torch.arange(k, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, seq_len, 1)
        )

        keys_scaled = base_keys.clone().detach().requires_grad_(True)
        scaled_output = module(keys_scaled, partition_indices)
        scaled_loss = (scaled_output * weight).sum()
        scaled_loss.backward()
        grad_scaled = keys_scaled.grad.detach().clone()

        keys_reference = base_keys.clone().detach().requires_grad_(True)
        reference_output = _reference_sparse_softmax(
            keys_reference, partition_indices, num_partitions
        )
        reference_loss = (reference_output * weight).sum()
        reference_loss.backward()
        grad_reference = keys_reference.grad.detach().clone()

        ratio = grad_scaled.abs().mean() / grad_reference.abs().mean()
        ratios.append(ratio.item())

    for ratio, k in zip(ratios, k_values):
        assert ratio == pytest.approx(1.0 / k, rel=1e-2, abs=1e-3)


def test_forward_remains_stable_for_large_values():
    torch.manual_seed(7)
    config = SSESparseSoftmaxConfig(
        num_partitions=6, state_rows_per_partition=64, k=3
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 4, 8) * 1e2
    partition_indices = torch.randint(0, config.num_partitions, (2, 4, config.k))

    output = module(keys, partition_indices)

    assert torch.all(torch.isfinite(output))
    assert torch.all(output >= 0)


def test_backward_produces_gradients_for_keys():
    torch.manual_seed(11)
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=3
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 4, requires_grad=True)
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k))
    weight = torch.randn(2, 3, config.num_partitions, 4)

    output = module(keys, partition_indices)
    loss = (output * weight).sum()
    loss.backward()

    assert keys.grad is not None
    assert torch.isfinite(keys.grad).all()
    assert keys.grad.abs().max() > 0


def test_backward_gradients_zero_for_unselected_partitions():
    torch.manual_seed(12)
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(1, 2, 3, requires_grad=True)
    partition_indices = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)

    output = module(keys, partition_indices)
    loss = output[:, :, 4, :].sum()  # Partition 4 is never selected
    loss.backward()

    assert torch.allclose(keys.grad, torch.zeros_like(keys.grad))


def test_autograd_gradcheck_passes():
    torch.manual_seed(13)
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config).to(dtype=torch.float64)

    keys = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    partition_indices = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)

    def func(input_keys: torch.Tensor) -> torch.Tensor:
        return module(input_keys, partition_indices)

    assert torch.autograd.gradcheck(func, (keys,), eps=1e-6, atol=1e-4)


def test_forward_matches_manual_dense_computation():
    keys = torch.tensor(
        [[[0.1, 0.2], [0.3, -0.4]]], dtype=torch.float64
    )
    partition_indices = torch.tensor([[[0, 2], [1, 2]]], dtype=torch.long)
    config = SSESparseSoftmaxConfig(
        num_partitions=3, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config).to(dtype=torch.float64)

    output = module(keys, partition_indices)
    reference = _reference_sparse_softmax(keys, partition_indices, config.num_partitions)

    assert torch.allclose(output, reference / config.k, atol=1e-6)


def test_integration_with_partition_selector_pipeline():
    torch.manual_seed(18)
    batch, seq_len, d_model = 2, 4, 6
    num_partitions = 5
    k = 3

    selector_config = SSEPartitionSelectorConfig(
        d_model=d_model, num_partitions=num_partitions, k=k
    )
    selector = SSEPartitionSelector(selector_config)

    softmax_config = SSESparseSoftmaxConfig(
        num_partitions=num_partitions, state_rows_per_partition=64, k=k
    )
    softmax = SSESparseSoftmax(softmax_config)

    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    partition_indices = selector(x.detach())
    output = softmax(x, partition_indices)

    assert output.shape == (batch, seq_len, num_partitions, d_model)
    assert torch.all(output >= 0)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_forward_remains_stable_for_small_values():
    torch.manual_seed(8)
    config = SSESparseSoftmaxConfig(
        num_partitions=5, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config)

    keys = torch.randn(2, 3, 6) * 1e-5
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k))

    output = module(keys, partition_indices)

    assert torch.all(torch.isfinite(output))
    assert torch.all(output >= 0)


def test_forward_supports_float64_precision():
    torch.manual_seed(9)
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config).to(dtype=torch.float64)

    keys = torch.randn(2, 3, 5, dtype=torch.float64)
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k))

    output = module(keys, partition_indices)

    assert output.dtype == torch.float64
    assert torch.all(torch.isfinite(output))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_supports_cuda_execution():
    torch.manual_seed(10)
    config = SSESparseSoftmaxConfig(
        num_partitions=4, state_rows_per_partition=64, k=2
    )
    module = SSESparseSoftmax(config).cuda()

    keys = torch.randn(2, 3, 6, device="cuda")
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, config.k), device="cuda")

    output = module(keys, partition_indices)

    assert output.is_cuda
    assert torch.all(torch.isfinite(output))


