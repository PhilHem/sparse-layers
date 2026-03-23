import pytest
import torch
from pydantic import ValidationError


def _build_config(**kwargs):
    from sparse_layers.modules import LinearAttentionConfig

    return LinearAttentionConfig(**kwargs)


def test_config_accepts_valid_values():
    config = _build_config(d_model=64, d_state=32, dropout=0.1)

    assert config.d_model == 64
    assert config.d_state == 32
    assert config.dropout == pytest.approx(0.1)


@pytest.mark.parametrize("d_model", [0, -1])
def test_config_requires_positive_d_model(d_model):
    with pytest.raises(ValidationError):
        _build_config(d_model=d_model, d_state=32, dropout=0.0)


@pytest.mark.parametrize("d_state", [0, -5])
def test_config_requires_positive_d_state(d_state):
    with pytest.raises(ValidationError):
        _build_config(d_model=64, d_state=d_state, dropout=0.0)


@pytest.mark.parametrize("dropout", [-0.01, 1.0, 2.0])
def test_config_requires_dropout_between_zero_and_one(dropout):
    with pytest.raises(ValidationError):
        _build_config(d_model=64, d_state=32, dropout=dropout)


def test_config_model_dump_matches_input():
    config = _build_config(d_model=32, d_state=16, dropout=0.25)

    assert config.model_dump() == {
        "d_model": 32,
        "d_state": 16,
        "dropout": pytest.approx(0.25),
    }


def test_config_is_immutable():
    config = _build_config(d_model=32, d_state=16, dropout=0.0)

    with pytest.raises(ValidationError):
        config.d_state = 64


def _build_module(**kwargs):
    from sparse_layers.modules import LinearAttention

    return LinearAttention(**kwargs)


def test_module_accepts_config_instance():
    config = _build_config(d_model=64, d_state=32, dropout=0.1)

    module = _build_module(config=config)

    assert module.config == config
    assert module.d_model == config.d_model
    assert module.d_state == config.d_state


def test_module_initializes_projection_layers_with_expected_dimensions():
    config = _build_config(d_model=48, d_state=12, dropout=0.2)

    module = _build_module(config=config)

    assert module.query.in_features == config.d_model
    assert module.query.out_features == config.d_state
    assert module.key.out_features == config.d_state
    assert module.value.out_features == config.d_model


def test_module_initializes_output_layer():
    config = _build_config(d_model=32, d_state=8, dropout=0.0)

    module = _build_module(config=config)

    assert module.output.in_features == config.d_model
    assert module.output.out_features == config.d_model


def test_module_parameters_are_trainable():
    config = _build_config(d_model=16, d_state=4, dropout=0.0)

    module = _build_module(config=config)

    parameters = list(module.parameters())

    assert parameters, "Expected module to expose trainable parameters"
    assert all(param.requires_grad for param in parameters)


def test_module_repr_includes_key_configuration_values():
    config = _build_config(d_model=40, d_state=10, dropout=0.15)

    module = _build_module(config=config)

    representation = repr(module)

    assert "d_model=40" in representation
    assert "d_state=10" in representation
    assert "dropout=0.15" in representation


def test_forward_requires_three_dimensional_inputs():
    config = _build_config(d_model=16, d_state=8, dropout=0.0)
    module = _build_module(config=config)
    inputs = torch.randn(config.d_model)

    with pytest.raises(ValueError, match="expected input of shape"):
        module(inputs)


def test_forward_requires_matching_feature_dimension():
    config = _build_config(d_model=16, d_state=8, dropout=0.0)
    module = _build_module(config=config)
    inputs = torch.randn(2, 3, config.d_model + 1)

    with pytest.raises(ValueError, match="expected last dimension"):
        module(inputs)


@pytest.mark.parametrize("batch_size, seq_len", [(1, 1), (2, 5), (4, 7)])
def test_forward_preserves_input_shape(batch_size, seq_len):
    config = _build_config(d_model=12, d_state=6, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(batch_size, seq_len, config.d_model)

    outputs = module(inputs)

    assert outputs.shape == (batch_size, seq_len, config.d_model)


def test_forward_matches_manual_outer_product_update():
    torch.manual_seed(0)
    config = _build_config(d_model=8, d_state=4, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(3, 5, config.d_model)

    with torch.no_grad():
        query = module.query(inputs)
        key = module.key(inputs)
        value = module.value(inputs)
        state = torch.zeros(inputs.size(0), config.d_state, config.d_model)
        expected_outputs = []
        for t in range(inputs.size(1)):
            update = key[:, t, :].unsqueeze(-1) * value[:, t, :].unsqueeze(-2)
            state = state + update
            output_t = torch.bmm(query[:, t, :].unsqueeze(1), state).squeeze(1)
            expected_outputs.append(output_t)
        expected = torch.stack(expected_outputs, dim=1)
        expected = module.output(expected)

    actual = module(inputs)

    assert torch.allclose(actual, expected)


def test_forward_is_deterministic_for_identical_inputs():
    torch.manual_seed(1)
    config = _build_config(d_model=6, d_state=3, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(2, 4, config.d_model)

    first = module(inputs)
    second = module(inputs)

    assert torch.allclose(first, second)


def test_forward_output_depends_on_previous_tokens():
    config = _build_config(d_model=4, d_state=2, dropout=0.0)
    module = _build_module(config=config)
    module.eval()

    base = torch.zeros(1, 3, config.d_model)
    base[0, 0, 0] = 1.0
    base[0, 1, 1] = 1.0

    outputs = module(base)

    assert not torch.allclose(outputs[:, 1, :], torch.zeros_like(outputs[:, 1, :]))
    # Third token should incorporate information from previous updates even if its input is zero
    assert torch.any(outputs[:, 2, :].abs() > 0)


def test_backward_computes_gradients_for_parameters():
    torch.manual_seed(2)
    config = _build_config(d_model=5, d_state=3, dropout=0.0)
    module = _build_module(config=config)
    inputs = torch.randn(4, 6, config.d_model, requires_grad=True)

    outputs = module(inputs)
    loss = outputs.pow(2).mean()
    loss.backward()

    grads = [param.grad for param in module.parameters() if param.requires_grad]

    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_gradient_magnitudes_within_reasonable_bounds():
    torch.manual_seed(3)
    config = _build_config(d_model=6, d_state=4, dropout=0.0)
    module = _build_module(config=config)
    inputs = torch.randn(8, 10, config.d_model, requires_grad=True)

    outputs = module(inputs)
    loss = outputs.mean()
    loss.backward()

    weight_grad = module.value.weight.grad
    assert weight_grad is not None
    mean_grad = weight_grad.abs().mean().item()
    assert 1e-6 <= mean_grad <= 1e-1


def test_forward_handles_long_sequence_without_nans():
    torch.manual_seed(4)
    config = _build_config(d_model=8, d_state=4, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(2, 256, config.d_model)

    outputs = module(inputs)

    assert torch.isfinite(outputs).all()


def test_backward_gradients_are_finite_for_long_sequences():
    torch.manual_seed(5)
    config = _build_config(d_model=7, d_state=5, dropout=0.0)
    module = _build_module(config=config)
    inputs = torch.randn(3, 128, config.d_model, requires_grad=True)

    outputs = module(inputs)
    scalar = outputs.sum()
    scalar.backward()

    assert torch.isfinite(inputs.grad).all()


def test_forward_supports_float64_inputs():
    torch.manual_seed(6)
    config = _build_config(d_model=5, d_state=3, dropout=0.0)
    module = _build_module(config=config).double()
    inputs = torch.randn(2, 4, config.d_model, dtype=torch.float64)

    outputs = module(inputs)

    assert outputs.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_runs_on_cuda():
    torch.manual_seed(7)
    config = _build_config(d_model=8, d_state=4, dropout=0.0)
    module = _build_module(config=config).cuda()
    inputs = torch.randn(2, 5, config.d_model, device="cuda")

    outputs = module(inputs)

    assert outputs.is_cuda


def test_dropout_introduces_stochasticity_in_training_mode():
    config = _build_config(d_model=10, d_state=5, dropout=0.3)
    module = _build_module(config=config)
    inputs = torch.randn(3, 6, config.d_model)

    module.train()
    torch.manual_seed(0)
    first = module(inputs)
    torch.manual_seed(1)
    second = module(inputs)

    assert not torch.allclose(first, second)


def test_dropout_disabled_in_eval_mode():
    config = _build_config(d_model=10, d_state=5, dropout=0.3)
    module = _build_module(config=config)
    inputs = torch.randn(3, 6, config.d_model)

    module.eval()
    torch.manual_seed(0)
    first = module(inputs)
    torch.manual_seed(1)
    second = module(inputs)

    assert torch.allclose(first, second)


def test_forward_handles_batch_size_one():
    config = _build_config(d_model=6, d_state=3, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(1, 4, config.d_model)

    outputs = module(inputs)

    assert outputs.shape == (1, 4, config.d_model)


def test_forward_handles_sequence_length_one():
    config = _build_config(d_model=6, d_state=3, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(5, 1, config.d_model)

    outputs = module(inputs)

    assert outputs.shape == (5, 1, config.d_model)


def test_forward_matches_vectorized_cumulative_computation():
    torch.manual_seed(123)
    config = _build_config(d_model=5, d_state=3, dropout=0.0)
    module = _build_module(config=config)
    module.eval()
    inputs = torch.randn(2, 6, config.d_model)

    query = module.query(inputs)
    key = module.key(inputs)
    value = module.value(inputs)

    outer = key.unsqueeze(-1) * value.unsqueeze(-2)
    cumulative_state = torch.cumsum(outer, dim=1)
    expected = torch.einsum("bsd,bsdm->bsm", query, cumulative_state)
    expected = module.output(expected)

    actual = module(inputs)

    assert torch.allclose(actual, expected)


def test_forward_matches_known_reference_output():
    config = _build_config(d_model=2, d_state=1, dropout=0.0)
    module = _build_module(config=config)
    module.eval()

    with torch.no_grad():
        module.query.weight.copy_(torch.ones_like(module.query.weight))
        module.query.bias.zero_()
        module.key.weight.copy_(torch.ones_like(module.key.weight))
        module.key.bias.zero_()
        module.value.weight.copy_(torch.eye(config.d_model))
        module.value.bias.zero_()
        module.output.weight.copy_(torch.eye(config.d_model))
        module.output.bias.zero_()

    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    outputs = module(inputs)

    expected = torch.tensor([[[9.0, 18.0], [168.0, 238.0]]])

    assert torch.allclose(outputs, expected)


