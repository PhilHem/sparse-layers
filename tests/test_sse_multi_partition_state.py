import pytest
import torch
from pydantic import ValidationError

from butterfly_layers.sse import SSEMultiPartitionStateConfig


def test_config_accepts_valid_values():
    config = SSEMultiPartitionStateConfig(
        num_partitions=4, c=128, d=768
    )

    assert config.num_partitions == 4
    assert config.c == 128
    assert config.d == 768


@pytest.mark.parametrize("num_partitions", [0, 1, -3])
def test_config_requires_num_partitions_greater_than_one(num_partitions):
    with pytest.raises(ValidationError):
        SSEMultiPartitionStateConfig(
            num_partitions=num_partitions, c=128, d=768
        )


@pytest.mark.parametrize("c", [0, -1, -16])
def test_config_requires_positive_c(c):
    with pytest.raises(ValidationError):
        SSEMultiPartitionStateConfig(
            num_partitions=4, c=c, d=768
        )


@pytest.mark.parametrize("d", [0, -1, -768])
def test_config_requires_positive_d(d):
    with pytest.raises(ValidationError):
        SSEMultiPartitionStateConfig(
            num_partitions=4, c=128, d=d
        )


def test_config_is_serializable():
    config = SSEMultiPartitionStateConfig(
        num_partitions=6, c=32, d=256
    )

    dumped = config.model_dump()

    assert dumped == {
        "num_partitions": 6,
        "c": 32,
        "d": 256,
    }


def test_config_is_immutable():
    config = SSEMultiPartitionStateConfig(
        num_partitions=4, c=128, d=768
    )

    with pytest.raises(ValidationError):
        config.c = 64


def _build_config(**overrides):
    defaults = dict(num_partitions=4, c=128, d=768)
    defaults.update(overrides)
    return SSEMultiPartitionStateConfig(**defaults)


def test_naive_module_initializes_with_config():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    assert module.config == config
    assert module.num_partitions == config.num_partitions
    assert module.c == config.c
    assert module.d == config.d


def test_naive_states_have_correct_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config(num_partitions=4, c=128, d=768)
    module = NaiveMultiPartitionState(config)

    assert len(module.states) == config.num_partitions
    for state in module.states:
        assert state.shape == (config.c, config.d)


def test_naive_states_are_not_trainable():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    for state in module.states:
        assert not state.requires_grad


def test_naive_update_requires_correct_partition_indices_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    keys = torch.randn(2, 3, 2, config.c)
    values = torch.randn(2, 3, config.d)

    # Should work with correct shape
    module.update(partition_indices, keys, values)

    # Should fail with wrong shape
    wrong_indices = torch.randint(0, config.num_partitions, (2, 2))
    with pytest.raises(ValueError, match="Expected partition_indices with shape"):
        module.update(wrong_indices, keys, values)


def test_naive_update_requires_correct_keys_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    values = torch.randn(2, 3, config.d)

    # Should fail with wrong shape
    wrong_keys = torch.randn(2, 3, config.c)
    with pytest.raises(ValueError, match="Expected keys with shape"):
        module.update(partition_indices, wrong_keys, values)


def test_naive_update_requires_correct_values_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    keys = torch.randn(2, 3, 2, config.c)

    # Should fail with wrong shape
    wrong_values = torch.randn(2, 3, 2, config.d)
    with pytest.raises(ValueError, match="Expected values with shape"):
        module.update(partition_indices, keys, wrong_values)


def test_naive_update_only_modifies_selected_partitions():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(0)
    config = _build_config(num_partitions=4, c=8, d=16)
    module = NaiveMultiPartitionState(config)

    # Store initial states
    initial_states = [state.data.clone() for state in module.states]

    # Update only partition 0
    partition_indices = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
    keys = torch.randn(1, 2, 2, config.c)
    values = torch.randn(1, 2, config.d)

    module.update(partition_indices, keys, values)

    # Partition 0 should be modified
    assert not torch.allclose(module.states[0].data, initial_states[0])

    # Partitions 1, 2, 3 should be unchanged
    for i in [1, 2, 3]:
        assert torch.allclose(module.states[i].data, initial_states[i])


def test_naive_update_non_selected_partitions_unchanged():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(1)
    config = _build_config(num_partitions=5, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # Store initial states
    initial_states = [state.data.clone() for state in module.states]

    # Update partitions 0 and 2
    partition_indices = torch.tensor([[[0, 2], [0, 2]]], dtype=torch.long)
    keys = torch.randn(1, 2, 2, config.c)
    values = torch.randn(1, 2, config.d)

    module.update(partition_indices, keys, values)

    # Partitions 0 and 2 should be modified
    assert not torch.allclose(module.states[0].data, initial_states[0])
    assert not torch.allclose(module.states[2].data, initial_states[2])

    # Partitions 1, 3, 4 should be unchanged
    for i in [1, 3, 4]:
        assert torch.allclose(module.states[i].data, initial_states[i])


def test_naive_update_handles_k_equals_one():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(2)
    config = _build_config(num_partitions=3, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.tensor([[[1], [2]]], dtype=torch.long)
    keys = torch.randn(1, 2, 1, config.c)
    values = torch.randn(1, 2, config.d)

    initial_state_1 = module.states[1].data.clone()
    initial_state_2 = module.states[2].data.clone()

    module.update(partition_indices, keys, values)

    # Both partitions should be updated
    assert not torch.allclose(module.states[1].data, initial_state_1)
    assert not torch.allclose(module.states[2].data, initial_state_2)


def test_naive_update_handles_multiple_selections_per_token():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(3)
    config = _build_config(num_partitions=4, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # Select partitions [0, 1] for first token, [2, 3] for second token
    partition_indices = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    keys = torch.randn(1, 2, 2, config.c)
    values = torch.randn(1, 2, config.d)

    initial_states = [state.data.clone() for state in module.states]

    module.update(partition_indices, keys, values)

    # All partitions should be updated
    for i in range(4):
        assert not torch.allclose(module.states[i].data, initial_states[i])


def test_naive_read_requires_correct_partition_indices_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    queries = torch.randn(2, 3, config.d)

    # Should work with correct shape
    output = module.read(partition_indices, queries)
    assert output.shape == (2, 3, config.d)

    # Should fail with wrong shape
    wrong_indices = torch.randint(0, config.num_partitions, (2, 2))
    with pytest.raises(ValueError, match="Expected partition_indices with shape"):
        module.read(wrong_indices, queries)


def test_naive_read_requires_correct_queries_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))

    # Should fail with wrong shape
    wrong_queries = torch.randn(2, 3, 2, config.d)
    with pytest.raises(ValueError, match="Expected queries with shape"):
        module.read(partition_indices, wrong_queries)


def test_naive_read_returns_correct_shape():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(4)
    config = _build_config()
    module = NaiveMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 5, 2))
    queries = torch.randn(2, 5, config.d)

    output = module.read(partition_indices, queries)

    assert output.shape == (2, 5, config.d)
    assert output.dtype == queries.dtype


def test_naive_read_sums_over_selected_partitions():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(5)
    config = _build_config(num_partitions=3, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # Initialize states with known values
    for i, state in enumerate(module.states):
        state.data.fill_(float(i + 1))

    # Select partitions [0, 1] for first token, [1, 2] for second token
    partition_indices = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    queries = torch.ones(1, 2, config.d)

    output = module.read(partition_indices, queries)

    # First token: q @ mean(S[0] + S[1]) across rows
    # S[0] is filled with 1.0 -> mean = 1.0, S[1] filled with 2.0 -> mean = 2.0
    # Sum of means: [3.0, 3.0, ...], multiply by q (all ones): [3.0, ...]
    # Second token: mean(S[1]) = 2.0, mean(S[2]) = 3.0 -> sum 5.0
    expected_first = torch.full((config.d,), 3.0)
    expected_second = torch.full((config.d,), 5.0)

    assert torch.allclose(output[0, 0], expected_first)
    assert torch.allclose(output[0, 1], expected_second)


def test_naive_read_matches_manual_computation():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(6)
    config = _build_config(num_partitions=2, c=3, d=4)
    module = NaiveMultiPartitionState(config)

    # Set specific state values
    module.states[0].data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    module.states[1].data = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]])

    partition_indices = torch.tensor([[[0, 1]]], dtype=torch.long)
    queries = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

    output = module.read(partition_indices, queries)

    # Manual computation: q @ mean(S[0] + S[1]) across rows
    # Column means: S[0] -> [5, 6, 7, 8], S[1] -> [50, 60, 70, 80]
    # Sum: [55, 66, 77, 88]
    expected = torch.tensor([[55.0, 66.0, 77.0, 88.0]])

    assert torch.allclose(output[0, 0], expected, atol=1e-5)


def test_naive_read_handles_k_equals_one():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(7)
    config = _build_config(num_partitions=3, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # Initialize state 1 with known values
    module.states[1].data.fill_(2.0)

    partition_indices = torch.tensor([[[1], [2]]], dtype=torch.long)
    queries = torch.ones(1, 2, config.d)

    output = module.read(partition_indices, queries)

    # Each query should read from a single partition
    assert output.shape == (1, 2, config.d)
    # First token reads from partition 1 (all 2.0s; mean=2.0), second from partition 2 (all 0.0s)
    assert torch.allclose(output[0, 0], torch.full((config.d,), 2.0))
    assert torch.allclose(output[0, 1], torch.zeros(config.d))


def test_naive_read_handles_k_equals_num_partitions():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(8)
    config = _build_config(num_partitions=3, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # Initialize all states with different values
    for i, state in enumerate(module.states):
        state.data.fill_(float(i + 1))

    # Select all partitions
    partition_indices = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    queries = torch.ones(1, 1, config.d)

    output = module.read(partition_indices, queries)

    # Should sum over partition means: 1 + 2 + 3 = 6 per element
    expected = torch.full((config.d,), 6.0)
    assert torch.allclose(output[0, 0], expected)


def test_naive_state_persists_across_multiple_updates():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(9)
    config = _build_config(num_partitions=3, c=4, d=8)
    module = NaiveMultiPartitionState(config)

    # First update
    partition_indices_1 = torch.tensor([[[0, 1]]], dtype=torch.long)
    keys_1 = torch.randn(1, 1, 2, config.c)
    values_1 = torch.randn(1, 1, config.d)
    module.update(partition_indices_1, keys_1, values_1)

    state_after_first = [state.data.clone() for state in module.states]

    # Second update to same partitions
    partition_indices_2 = torch.tensor([[[0, 1]]], dtype=torch.long)
    keys_2 = torch.randn(1, 1, 2, config.c)
    values_2 = torch.randn(1, 1, config.d)
    module.update(partition_indices_2, keys_2, values_2)

    # States should have changed (accumulated)
    assert not torch.allclose(module.states[0].data, state_after_first[0])
    assert not torch.allclose(module.states[1].data, state_after_first[1])
    # Partition 2 should be unchanged
    assert torch.allclose(module.states[2].data, state_after_first[2])


def test_naive_reset_state_clears_all_partitions():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(10)
    config = _build_config()
    module = NaiveMultiPartitionState(config)

    # Update some partitions
    partition_indices = torch.tensor([[[0, 2], [1, 2]]], dtype=torch.long)
    keys = torch.randn(1, 2, 2, config.c)
    values = torch.randn(1, 2, config.d)
    module.update(partition_indices, keys, values)

    # Verify states are non-zero
    assert torch.any(module.states[0].data != 0)
    assert torch.any(module.states[1].data != 0)
    assert torch.any(module.states[2].data != 0)

    # Reset
    module.reset_state()

    # All states should be zero
    for state in module.states:
        assert torch.allclose(state.data, torch.zeros_like(state.data))


def test_naive_state_norms_do_not_collapse():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(11)
    config = _build_config(num_partitions=4, c=8, d=16)
    module = NaiveMultiPartitionState(config)

    # Multiple updates
    for _ in range(10):
        partition_indices = torch.randint(0, config.num_partitions, (2, 5, 2))
        keys = torch.randn(2, 5, 2, config.c)
        values = torch.randn(2, 5, config.d)
        module.update(partition_indices, keys, values)

    # Check that state norms are reasonable (not collapsed to zero or exploded)
    for state in module.states:
        norm = state.data.norm().item()
        assert norm > 0.0
        assert norm < 1e6  # Not exploded


def test_naive_update_incremental_matches_batch_computation():
    from butterfly_layers.sse import NaiveMultiPartitionState

    torch.manual_seed(12)
    config = _build_config(num_partitions=3, c=4, d=8)
    
    # Incremental updates
    module_inc = NaiveMultiPartitionState(config)
    partition_indices = torch.tensor([[[0], [1], [2]]], dtype=torch.long)
    keys = torch.randn(1, 3, 1, config.c)
    values = torch.randn(1, 3, config.d)
    
    for t in range(3):
        module_inc.update(
            partition_indices[:, t:t+1, :],
            keys[:, t:t+1, :, :],
            values[:, t:t+1, :]
        )

    # Batch update
    module_batch = NaiveMultiPartitionState(config)
    module_batch.update(partition_indices, keys, values)

    # States should match
    for i in range(config.num_partitions):
        assert torch.allclose(module_inc.states[i].data, module_batch.states[i].data)


def test_optimized_module_initializes_with_config():
    from butterfly_layers.sse import SSEMultiPartitionState

    config = _build_config()
    module = SSEMultiPartitionState(config)

    assert module.config == config
    assert module.num_partitions == config.num_partitions
    assert module.c == config.c
    assert module.d == config.d


def test_optimized_states_have_correct_shape():
    from butterfly_layers.sse import SSEMultiPartitionState

    config = _build_config(num_partitions=4, c=128, d=768)
    module = SSEMultiPartitionState(config)

    assert module.states.shape == (config.num_partitions, config.c, config.d)
    assert not module.states.requires_grad


def test_optimized_update_matches_naive_outputs():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(13)
    config = _build_config(num_partitions=4, c=8, d=16)
    
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (2, 5, 2))
    keys = torch.randn(2, 5, 2, config.c)
    values = torch.randn(2, 5, config.d)

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    # Compare states
    for i in range(config.num_partitions):
        assert torch.allclose(naive.states[i].data, optimized.states[i], atol=1e-6)


def test_optimized_read_matches_naive_outputs():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(14)
    config = _build_config(num_partitions=3, c=6, d=12)
    
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    # Update both with same data
    partition_indices = torch.randint(0, config.num_partitions, (2, 4, 2))
    keys = torch.randn(2, 4, 2, config.c)
    values = torch.randn(2, 4, config.d)

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    # Read from both
    queries = torch.randn(2, 4, config.d)
    naive_output = naive.read(partition_indices, queries)
    optimized_output = optimized.read(partition_indices, queries)

    assert torch.allclose(naive_output, optimized_output, atol=1e-6)


def test_optimized_update_only_modifies_selected_partitions():
    from butterfly_layers.sse import SSEMultiPartitionState

    torch.manual_seed(15)
    config = _build_config(num_partitions=5, c=4, d=8)
    module = SSEMultiPartitionState(config)

    initial_states = module.states.clone()

    partition_indices = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.long)
    keys = torch.randn(1, 2, 2, config.c)
    values = torch.randn(1, 2, config.d)

    module.update(partition_indices, keys, values)

    # Selected partitions should be modified
    assert not torch.allclose(module.states[0], initial_states[0])
    assert not torch.allclose(module.states[1], initial_states[1])
    assert not torch.allclose(module.states[2], initial_states[2])
    assert not torch.allclose(module.states[3], initial_states[3])

    # Non-selected partition should be unchanged
    assert torch.allclose(module.states[4], initial_states[4])


def test_optimized_memory_isolation_verified():
    from butterfly_layers.sse import SSEMultiPartitionState

    torch.manual_seed(16)
    config = _build_config(num_partitions=4, c=4, d=8)
    module = SSEMultiPartitionState(config)

    # Update partition 0
    partition_indices_0 = torch.tensor([[[0, 0]]], dtype=torch.long)
    keys_0 = torch.randn(1, 1, 2, config.c)
    values_0 = torch.randn(1, 1, config.d)
    state_0_before = module.states[0].clone()
    state_1_before = module.states[1].clone()

    module.update(partition_indices_0, keys_0, values_0)

    # Only partition 0 should change
    assert not torch.allclose(module.states[0], state_0_before)
    assert torch.allclose(module.states[1], state_1_before)
    assert torch.allclose(module.states[2], torch.zeros_like(module.states[2]))
    assert torch.allclose(module.states[3], torch.zeros_like(module.states[3]))


def test_optimized_reset_state_matches_naive():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(17)
    config = _build_config()
    
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    # Update both
    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    keys = torch.randn(2, 3, 2, config.c)
    values = torch.randn(2, 3, config.d)

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    # Reset both
    naive.reset_state()
    optimized.reset_state()

    # All states should be zero
    for i in range(config.num_partitions):
        assert torch.allclose(naive.states[i].data, torch.zeros_like(naive.states[i].data))
        assert torch.allclose(optimized.states[i], torch.zeros_like(optimized.states[i]))


def test_both_implementations_handle_zero_sequence_length():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    config = _build_config()
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    # Zero sequence length
    partition_indices = torch.randint(0, config.num_partitions, (2, 0, 2))
    keys = torch.randn(2, 0, 2, config.c)
    values = torch.randn(2, 0, config.d)
    queries = torch.randn(2, 0, config.d)

    # Should not raise errors
    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    naive_output = naive.read(partition_indices, queries)
    optimized_output = optimized.read(partition_indices, queries)

    assert naive_output.shape == (2, 0, config.d)
    assert optimized_output.shape == (2, 0, config.d)


def test_both_implementations_handle_single_batch_token():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(18)
    config = _build_config()
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    partition_indices = torch.randint(0, config.num_partitions, (1, 1, 2))
    keys = torch.randn(1, 1, 2, config.c)
    values = torch.randn(1, 1, config.d)
    queries = torch.randn(1, 1, config.d)

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    naive_output = naive.read(partition_indices, queries)
    optimized_output = optimized.read(partition_indices, queries)

    assert naive_output.shape == (1, 1, config.d)
    assert optimized_output.shape == (1, 1, config.d)
    assert torch.allclose(naive_output, optimized_output, atol=1e-6)


def test_both_implementations_support_float64():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(19)
    config = _build_config()
    naive = NaiveMultiPartitionState(config)
    optimized = SSEMultiPartitionState(config)

    # Convert to float64
    for state in naive.states:
        state.data = state.data.to(dtype=torch.float64)
    optimized.states.data = optimized.states.data.to(dtype=torch.float64)

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2))
    keys = torch.randn(2, 3, 2, config.c, dtype=torch.float64)
    values = torch.randn(2, 3, config.d, dtype=torch.float64)
    queries = torch.randn(2, 3, config.d, dtype=torch.float64)

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    naive_output = naive.read(partition_indices, queries)
    optimized_output = optimized.read(partition_indices, queries)

    assert naive_output.dtype == torch.float64
    assert optimized_output.dtype == torch.float64
    assert torch.allclose(naive_output, optimized_output, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_both_implementations_support_cuda():
    from butterfly_layers.sse import (
        NaiveMultiPartitionState,
        SSEMultiPartitionState,
    )

    torch.manual_seed(20)
    config = _build_config()
    naive = NaiveMultiPartitionState(config).cuda()
    optimized = SSEMultiPartitionState(config).cuda()

    partition_indices = torch.randint(0, config.num_partitions, (2, 3, 2), device="cuda")
    keys = torch.randn(2, 3, 2, config.c, device="cuda")
    values = torch.randn(2, 3, config.d, device="cuda")
    queries = torch.randn(2, 3, config.d, device="cuda")

    naive.update(partition_indices, keys, values)
    optimized.update(partition_indices, keys, values)

    naive_output = naive.read(partition_indices, queries)
    optimized_output = optimized.read(partition_indices, queries)

    assert naive_output.is_cuda
    assert optimized_output.is_cuda
    assert torch.allclose(naive_output, optimized_output, atol=1e-6)

