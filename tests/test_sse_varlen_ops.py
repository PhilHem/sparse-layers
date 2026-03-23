import pytest
import torch
from pydantic import ValidationError

from sparse_layers.ops import SSEVarlenOps, SSEVarlenOpsConfig


def _build_config(**overrides) -> SSEVarlenOpsConfig:
    defaults = dict(d_model=64, num_partitions=8, k=4)
    defaults.update(overrides)
    return SSEVarlenOpsConfig(**defaults)


def test_varlen_config_accepts_valid_values():
    config = _build_config(return_inverse=False)

    assert config.d_model == 64
    assert config.num_partitions == 8
    assert config.k == 4
    assert config.return_inverse is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 0),
        ("d_model", -1),
        ("num_partitions", 0),
        ("num_partitions", 1),
        ("num_partitions", -3),
        ("k", 0),
        ("k", -2),
    ],
)
def test_varlen_config_rejects_non_positive_fields(field, value):
    kwargs = dict(d_model=64, num_partitions=8, k=4)
    kwargs[field] = value

    with pytest.raises(ValidationError):
        SSEVarlenOpsConfig(**kwargs)


def test_varlen_config_requires_k_within_partitions():
    with pytest.raises(ValidationError):
        _build_config(num_partitions=4, k=5)


def test_varlen_config_is_immutable():
    config = _build_config()

    with pytest.raises(ValidationError):
        config.k = 3


def test_varlen_config_enables_optional_inverse_flag():
    config = _build_config(return_inverse=True)

    assert config.return_inverse is True


# -----------------------------------------------------------------------------
# Naive reference implementation (RED phase for forward)
# -----------------------------------------------------------------------------


def _naive_varlen(
    x: torch.Tensor,
    partition_indices: torch.Tensor,
    num_partitions: int,
    *,
    return_inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch, seq_len, d_model = x.shape
    k = partition_indices.shape[-1]

    packed_tokens: list[torch.Tensor] = []
    cu_seqlens = [0]
    inverse = (
        torch.full(
            (batch, seq_len, k),
            -1,
            dtype=torch.long,
            device=x.device,
        )
        if return_inverse
        else None
    )

    offset = 0
    for b in range(batch):
        for partition in range(num_partitions):
            count = 0
            for t in range(seq_len):
                for i in range(k):
                    if partition_indices[b, t, i].item() == partition:
                        packed_tokens.append(x[b, t])
                        if inverse is not None:
                            inverse[b, t, i] = offset + count
                        count += 1
            offset += count
            cu_seqlens.append(offset)

    if packed_tokens:
        packed = torch.stack(packed_tokens, dim=0)
    else:
        packed = x.new_zeros((0, d_model))

    cu_tensor = torch.tensor(
        cu_seqlens,
        dtype=torch.int32,
        device=x.device,
    )

    return packed, cu_tensor, inverse


def _build_module(**config_overrides) -> SSEVarlenOps:
    config = _build_config(**config_overrides)
    return SSEVarlenOps(config)


# -----------------------------------------------------------------------------
# Module behaviour tests (RED phase)
# -----------------------------------------------------------------------------


def test_varlen_ops_matches_naive_reordering():
    torch.manual_seed(0)
    module = _build_module(num_partitions=6, k=3, return_inverse=True)
    x = torch.randn(2, 5, module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (2, 5, module.k), dtype=torch.long
    )

    expected_tokens, expected_cu, expected_inverse = _naive_varlen(
        x, partition_indices, module.num_partitions, return_inverse=True
    )
    actual_tokens, actual_cu, actual_inverse = module(x, partition_indices)

    assert torch.allclose(actual_tokens, expected_tokens)
    assert torch.equal(actual_cu, expected_cu)
    assert torch.equal(actual_inverse, expected_inverse)


def test_varlen_ops_returns_none_inverse_when_disabled():
    module = _build_module(return_inverse=False)
    x = torch.randn(1, 4, module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (1, 4, module.k), dtype=torch.long
    )

    _, _, inverse = module(x, partition_indices)

    assert inverse is None


def test_varlen_ops_computes_cu_seqlens():
    torch.manual_seed(1)
    module = _build_module(num_partitions=4, k=2, return_inverse=True)
    x = torch.randn(3, 6, module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (3, 6, module.k), dtype=torch.long
    )

    packed, cu_seqlens, _ = module(x, partition_indices)

    assert cu_seqlens.shape == (module.config.num_partitions * x.size(0) + 1,)
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0].item() == 0
    assert cu_seqlens[-1].item() == packed.shape[0]
    assert torch.all(cu_seqlens[1:] >= cu_seqlens[:-1])


def test_varlen_ops_handles_empty_sequence():
    module = _build_module(num_partitions=5, k=3, return_inverse=True)
    x = torch.randn(2, 0, module.config.d_model)
    partition_indices = torch.empty(2, 0, module.k, dtype=torch.long)

    packed, cu_seqlens, inverse = module(x, partition_indices)

    assert packed.shape == (0, module.config.d_model)
    assert cu_seqlens.shape == (module.config.num_partitions * x.size(0) + 1,)
    assert torch.all(cu_seqlens == 0)
    assert torch.all(inverse == -1)


def test_varlen_ops_handles_partitions_without_tokens():
    module = _build_module(num_partitions=3, k=2, return_inverse=True)
    x = torch.randn(1, 3, module.config.d_model)
    partition_indices = torch.tensor([[[0, 0], [1, 1], [1, 1]]], dtype=torch.long)

    packed, cu_seqlens, inverse = module(x, partition_indices)
    expected_packed, expected_cu, expected_inverse = _naive_varlen(
        x, partition_indices, module.num_partitions, return_inverse=True
    )

    assert torch.equal(cu_seqlens, expected_cu)
    assert torch.equal(packed, expected_packed)
    assert torch.equal(inverse, expected_inverse)
    # Final partition receives no tokens; CU entry should remain unchanged
    assert cu_seqlens[-1].item() == cu_seqlens[-2].item()


def test_varlen_ops_preserves_gradients():
    torch.manual_seed(2)
    module = _build_module(num_partitions=4, k=2, return_inverse=False)
    x = torch.randn(3, 5, module.config.d_model, requires_grad=True)
    partition_indices = torch.randint(
        0, module.num_partitions, (3, 5, module.k), dtype=torch.long
    )

    packed, _, _ = module(x, partition_indices)
    loss = packed.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_varlen_ops_validates_input_dimensions():
    module = _build_module()
    x = torch.randn(module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (module.k,), dtype=torch.long
    )

    with pytest.raises(ValueError, match="expected input of shape"):
        module(x, partition_indices)


def test_varlen_ops_validates_partition_indices_dimensions():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (2, module.k), dtype=torch.long
    )

    with pytest.raises(ValueError, match="expected partition_indices with shape"):
        module(x, partition_indices)


def test_varlen_ops_validates_feature_dimension():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model + 1)
    partition_indices = torch.randint(
        0, module.num_partitions, (2, 3, module.k), dtype=torch.long
    )

    with pytest.raises(ValueError, match="expected last dimension"):
        module(x, partition_indices)


def test_varlen_ops_validates_partition_k_matches_config():
    module = _build_module(k=2)
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randint(
        0, module.num_partitions, (2, 3, module.k + 1), dtype=torch.long
    )

    with pytest.raises(ValueError, match="expected partition_indices last dimension"):
        module(x, partition_indices)


def test_varlen_ops_requires_long_partition_indices():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.randn(2, 3, module.k)

    with pytest.raises(ValueError, match="partition_indices must have dtype torch.long"):
        module(x, partition_indices)


def test_varlen_ops_validates_partition_range():
    module = _build_module()
    x = torch.randn(2, 3, module.config.d_model)
    partition_indices = torch.full((2, 3, module.k), module.config.num_partitions)

    with pytest.raises(ValueError, match="values outside valid partition range"):
        module(x, partition_indices)

