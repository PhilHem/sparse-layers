import pytest
import torch
from pydantic import ValidationError

from sparse_layers.sse import SSEPartitionSelector, SSEPartitionSelectorConfig


def test_config_accepts_valid_values():
    config = SSEPartitionSelectorConfig(d_model=64, num_partitions=8, k=4)

    assert config.d_model == 64
    assert config.num_partitions == 8
    assert config.k == 4


@pytest.mark.parametrize("d_model", [0, -1])
def test_config_requires_positive_d_model(d_model):
    with pytest.raises(ValidationError):
        SSEPartitionSelectorConfig(d_model=d_model, num_partitions=4, k=2)


@pytest.mark.parametrize("num_partitions", [0, 1, -3])
def test_config_requires_num_partitions_greater_than_one(num_partitions):
    with pytest.raises(ValidationError):
        SSEPartitionSelectorConfig(d_model=32, num_partitions=num_partitions, k=2)


@pytest.mark.parametrize("k", [0, -2])
def test_config_requires_positive_k(k):
    with pytest.raises(ValidationError):
        SSEPartitionSelectorConfig(d_model=32, num_partitions=4, k=k)


def test_config_requires_k_less_or_equal_num_partitions():
    with pytest.raises(ValidationError):
        SSEPartitionSelectorConfig(d_model=32, num_partitions=4, k=5)


def test_config_is_serializable():
    config = SSEPartitionSelectorConfig(d_model=128, num_partitions=16, k=8)

    dumped = config.model_dump()

    assert dumped == {"d_model": 128, "num_partitions": 16, "k": 8}


def test_config_is_immutable():
    config = SSEPartitionSelectorConfig(d_model=32, num_partitions=4, k=2)

    with pytest.raises(ValidationError):
        config.k = 1


def test_selector_initializes_with_config():
    config = SSEPartitionSelectorConfig(d_model=32, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)

    assert selector.config == config
    assert selector.k == config.k
    assert selector.num_partitions == config.num_partitions


def test_selector_creates_linear_projection_layer():
    config = SSEPartitionSelectorConfig(d_model=64, num_partitions=8, k=4)
    selector = SSEPartitionSelector(config)

    assert selector.linear.in_features == config.d_model
    assert selector.linear.out_features == config.num_partitions


def test_selector_parameters_are_trainable():
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)

    parameters = list(selector.parameters())

    assert parameters, "Selector should expose trainable parameters"
    assert all(param.requires_grad for param in parameters)


def test_selector_repr_includes_key_attributes():
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)

    representation = repr(selector)

    assert "k=2" in representation
    assert "num_partitions=4" in representation
    assert "d_model=16" in representation


def test_forward_returns_top_k_indices_with_expected_shape():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=32, num_partitions=6, k=3)
    selector = SSEPartitionSelector(config)
    x = torch.randn(2, 5, config.d_model)

    indices = selector(x)

    assert indices.shape == (2, 5, config.k)
    assert indices.dtype == torch.int64
    assert indices.min() >= 0
    assert indices.max() < config.num_partitions


def test_forward_handles_k_equals_one():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=4, k=1)
    selector = SSEPartitionSelector(config)
    x = torch.randn(4, 7, config.d_model)

    indices = selector(x)

    assert indices.shape == (4, 7, 1)


def test_forward_handles_k_equals_num_partitions():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=3, k=3)
    selector = SSEPartitionSelector(config)
    x = torch.randn(3, 2, config.d_model)

    indices = selector(x)

    assert indices.shape == (3, 2, config.k)
    assert torch.all(indices < config.num_partitions)

    sorted_indices = indices.sort(dim=-1).values
    expected = torch.arange(config.num_partitions, device=indices.device)
    expected = expected.expand_as(sorted_indices)
    assert torch.equal(sorted_indices, expected)


def test_forward_raises_for_invalid_rank():
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(16, config.d_model)

    with pytest.raises(ValueError):
        selector(x)


def test_forward_raises_for_mismatched_feature_dimension():
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(2, 3, config.d_model + 1)

    with pytest.raises(ValueError):
        selector(x)


def test_backward_computes_gradients_for_linear_parameters():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=8, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(6, 5, config.d_model)

    bias_scores = selector.linear(x)
    loss = bias_scores.sum()
    loss.backward()

    gradients = [param.grad for param in selector.linear.parameters()]

    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_backward_gradients_have_reasonable_magnitude():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=8, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(10, 12, config.d_model)

    bias_scores = selector.linear(x)
    loss = bias_scores.mean()
    loss.backward()

    weight_grad = selector.linear.weight.grad

    assert weight_grad is not None
    mean_grad = weight_grad.abs().mean().item()
    assert 1e-6 <= mean_grad <= 1e-1


def test_partition_selection_distribution_is_well_spread():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=6, k=2)
    selector = SSEPartitionSelector(config)
    counts = torch.zeros(config.num_partitions, dtype=torch.int64)

    for _ in range(64):
        x = torch.randn(8, 10, config.d_model)
        selections = selector(x)
        counts += torch.bincount(
            selections.flatten(), minlength=config.num_partitions
        )

    assert torch.all(counts > 0)
    imbalance = counts.float() / counts.sum()
    assert imbalance.max() - imbalance.min() < 0.5


def test_partition_selection_avoids_dead_partitions():
    torch.manual_seed(1)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=5, k=2)
    selector = SSEPartitionSelector(config)
    seen = torch.zeros(config.num_partitions, dtype=torch.bool)

    for _ in range(50):
        x = torch.randn(4, 6, config.d_model)
        selections = selector(x)
        seen[selections.flatten().unique()] = True

    assert seen.all()


def test_bias_gradients_maintain_stable_statistics():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=12, num_partitions=6, k=3)
    selector = SSEPartitionSelector(config)
    mean_magnitudes = []

    for _ in range(10):
        selector.linear.zero_grad(set_to_none=True)
        x = torch.randn(3, 7, config.d_model)
        bias_scores = selector.linear(x)
        loss = bias_scores.pow(2).mean()
        loss.backward()
        grad = selector.linear.bias.grad
        assert grad is not None
        mean_magnitudes.append(grad.abs().mean().item())

    min_mean = min(mean_magnitudes)
    max_mean = max(mean_magnitudes)
    assert min_mean >= 1e-7
    assert max_mean <= 1e-1
    assert max_mean / min_mean <= 50


def test_forward_supports_float64_inputs():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=10, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    selector = selector.double()
    x = torch.randn(3, 5, config.d_model, dtype=torch.float64)

    indices = selector(x)

    assert indices.dtype == torch.int64


def test_forward_handles_batch_size_one():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=12, num_partitions=5, k=3)
    selector = SSEPartitionSelector(config)
    x = torch.randn(1, 7, config.d_model)

    indices = selector(x)

    assert indices.shape == (1, 7, config.k)


def test_forward_handles_sequence_length_one():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=14, num_partitions=6, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(4, 1, config.d_model)

    indices = selector(x)

    assert indices.shape == (4, 1, config.k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_runs_on_cuda():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=8, k=4)
    selector = SSEPartitionSelector(config).cuda()
    x = torch.randn(2, 9, config.d_model, device="cuda")

    indices = selector(x)

    assert indices.is_cuda


def test_forward_handles_long_sequence_length():
    torch.manual_seed(0)
    config = SSEPartitionSelectorConfig(d_model=8, num_partitions=4, k=2)
    selector = SSEPartitionSelector(config)
    x = torch.randn(2, 512, config.d_model)

    indices = selector(x)

    assert indices.shape == (2, 512, config.k)


def test_selector_returns_actual_top_k_highest_scoring_partitions():
    torch.manual_seed(42)
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=8, k=3)
    selector = SSEPartitionSelector(config)
    x = torch.randn(4, 6, config.d_model)

    indices = selector(x)
    bias_scores = selector.linear(x)
    expected = bias_scores.topk(config.k, dim=-1).indices

    assert torch.equal(indices, expected)


def test_partition_selections_independent_across_sequence_positions():
    torch.manual_seed(42)
    config = SSEPartitionSelectorConfig(d_model=12, num_partitions=6, k=2)
    selector = SSEPartitionSelector(config)

    x = torch.randn(2, 3, config.d_model)
    x[:, 2, :] = x[:, 0, :]

    indices = selector(x)

    assert torch.equal(indices[:, 0, :], indices[:, 2, :])
    assert not torch.all(indices[:, 0, :] == indices[:, 1, :])


def test_forward_deterministic_with_same_input():
    config = SSEPartitionSelectorConfig(d_model=16, num_partitions=8, k=4)
    selector = SSEPartitionSelector(config)
    x = torch.randn(3, 5, config.d_model)

    first = selector(x)
    second = selector(x)

    assert torch.equal(first, second)


def test_bias_scores_reasonable_magnitude_for_softmax():
    torch.manual_seed(42)
    config = SSEPartitionSelectorConfig(d_model=32, num_partitions=8, k=3)
    selector = SSEPartitionSelector(config)
    x = torch.randn(10, 20, config.d_model)

    bias_scores = selector.linear(x)

    mean_abs = bias_scores.abs().mean().item()
    max_abs = bias_scores.abs().max().item()
    std = bias_scores.std().item()

    assert 0.01 < mean_abs < 100, mean_abs
    assert max_abs < 1000, max_abs
    assert std > 0.01, std

