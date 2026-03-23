import pytest
import torch
from pydantic import ValidationError

from butterfly_layers.sse import SSEMaskingOps, SSEMaskingOpsConfig


def test_masking_config_accepts_valid_values():
    config = SSEMaskingOpsConfig(d_model=64, num_partitions=8, k=4)

    assert config.d_model == 64
    assert config.num_partitions == 8
    assert config.k == 4


@pytest.mark.parametrize("field, value", [
    ("d_model", 0),
    ("d_model", -1),
    ("num_partitions", 0),
    ("num_partitions", 1),
    ("num_partitions", -3),
    ("k", 0),
    ("k", -5),
])
def test_masking_config_rejects_non_positive_fields(field, value):
    kwargs = dict(d_model=32, num_partitions=4, k=2)
    kwargs[field] = value

    with pytest.raises(ValidationError):
        SSEMaskingOpsConfig(**kwargs)


def test_masking_config_requires_k_within_partitions():
    with pytest.raises(ValidationError):
        SSEMaskingOpsConfig(d_model=32, num_partitions=4, k=5)


def test_masking_config_is_immutable():
    config = SSEMaskingOpsConfig(d_model=32, num_partitions=4, k=2)

    with pytest.raises(ValidationError):
        config.k = 1


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _build_config(**overrides) -> SSEMaskingOpsConfig:
    defaults = dict(d_model=32, num_partitions=6, k=3)
    defaults.update(overrides)
    return SSEMaskingOpsConfig(**defaults)


def _naive_masking(x: torch.Tensor, partition_indices: torch.Tensor, num_partitions: int) -> torch.Tensor:
    batch, seq_len, d_model = x.shape
    output = torch.zeros(batch, seq_len, num_partitions, d_model, dtype=x.dtype, device=x.device)

    for b in range(batch):
        for t in range(seq_len):
            token = x[b, t]
            for partition_idx in partition_indices[b, t]:
                output[b, t, partition_idx] = token

    return output


def _build_module(**config_overrides) -> SSEMaskingOps:
    config = _build_config(**config_overrides)
    return SSEMaskingOps(config)


# -----------------------------------------------------------------------------
# Module behaviour tests (RED phase)
# -----------------------------------------------------------------------------


def test_masking_ops_matches_naive_reference():
    torch.manual_seed(0)
    module = _build_module(d_model=16, num_partitions=8, k=3)
    x = torch.randn(2, 5, module.config.d_model)
    partition_indices = torch.randint(0, module.num_partitions, (2, 5, module.k))

    expected = _naive_masking(x, partition_indices, module.num_partitions)
    actual = module(x, partition_indices)

    assert torch.allclose(actual, expected)


def test_masking_ops_masks_non_selected_partitions():
    torch.manual_seed(1)
    module = _build_module(num_partitions=6, k=2)
    x = torch.randn(1, 4, module.config.d_model)
    partition_indices = torch.tensor([[[0, 1], [2, 3], [4, 5], [0, 5]]], dtype=torch.long)

    output = module(x, partition_indices)

    selected_mask = torch.zeros(1, 4, module.num_partitions, dtype=torch.bool)
    selected_mask.scatter_(2, partition_indices, True)
    assert torch.all(output.masked_select(~selected_mask.unsqueeze(-1)).abs() < 1e-8)


@pytest.mark.parametrize("seq_len", [0, 1, 16, 64, 128, 256])
def test_masking_ops_handles_various_sequence_lengths(seq_len):
    torch.manual_seed(2)
    module = _build_module(d_model=8, num_partitions=5, k=2)
    x = torch.randn(3, seq_len, module.config.d_model)
    partition_indices = torch.randint(0, module.num_partitions, (3, seq_len, module.k)) if seq_len > 0 else torch.empty(3, 0, module.k, dtype=torch.long)

    output = module(x, partition_indices)

    assert output.shape == (3, seq_len, module.num_partitions, module.config.d_model)
    if seq_len == 0:
        assert output.numel() == 0


def test_masking_ops_preserves_gradients():
    torch.manual_seed(3)
    module = _build_module(d_model=10, num_partitions=7, k=3)
    x = torch.randn(4, 6, module.config.d_model, requires_grad=True)
    partition_indices = torch.randint(0, module.num_partitions, (4, 6, module.k))

    output = module(x, partition_indices)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_masking_ops_validates_input_dimensions():
    module = _build_module()
    x = torch.randn(module.config.d_model)
    partition_indices = torch.randint(0, module.num_partitions, (module.k,))

    with pytest.raises(ValueError, match="expected input of shape"):
        module(x, partition_indices)


def test_masking_ops_validates_partition_indices_dimensions():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randint(0, module.num_partitions, (2, module.k))

    with pytest.raises(ValueError, match="expected partition_indices with shape"):
        module(x, partition_indices)


def test_masking_ops_validates_feature_dimension():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model + 1)
    partition_indices = torch.randint(0, module.num_partitions, (2, 3, module.k))

    with pytest.raises(ValueError, match="expected last dimension"):
        module(x, partition_indices)


def test_masking_ops_validates_partition_k_matches_config():
    module = _build_module(k=2)
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randint(0, module.num_partitions, (2, 3, module.k + 1))

    with pytest.raises(ValueError, match="expected partition_indices last dimension"):
        module(x, partition_indices)


def test_masking_ops_requires_long_partition_indices():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randn(2, 3, module.k)

    with pytest.raises(ValueError, match="partition_indices must have dtype torch.long"):
        module(x, partition_indices)


def test_masking_ops_handles_duplicate_partition_selections():
    module = _build_module(num_partitions=4, k=3)
    x = torch.ones(1, 2, module.config.d_model)
    partition_indices = torch.tensor([[[1, 1, 2], [0, 0, 0]]], dtype=torch.long)

    output = module(x, partition_indices)

    assert torch.allclose(output[0, 0, 1], x[0, 0])
    assert torch.allclose(output[0, 0, 2], x[0, 0])
    assert torch.all(output[0, 0, 0] == 0)
    assert torch.allclose(output[0, 1, 0], x[0, 1])
