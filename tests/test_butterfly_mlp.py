import pytest
import torch

from sparse_layers import ButterflyLinear, ButterflyMLP, SimpleMLP


def test_initialization_creates_sparse_layers():
    model = ButterflyMLP(input_dim=8, hidden_dims=[8, 8], output_dim=8)

    sparse_layers = [module for module in model.modules() if isinstance(module, ButterflyLinear)]

    assert len(sparse_layers) == 3
    assert all(layer.in_features == layer.out_features == 8 for layer in sparse_layers)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_pass_has_correct_shape(batch_size):
    model = ButterflyMLP(input_dim=4, hidden_dims=[4], output_dim=4)
    x = torch.randn(batch_size, 4)

    output = model(x)

    assert output.shape == (batch_size, 4)


def test_invalid_dimensions_raise_value_error():
    with pytest.raises(ValueError):
        ButterflyMLP(input_dim=6, hidden_dims=[6], output_dim=6)


def test_parameter_count_is_smaller_than_simple_mlp():
    butterfly = ButterflyMLP(input_dim=8, hidden_dims=[8, 8], output_dim=8)
    simple = SimpleMLP(input_dim=8, hidden_dims=[8, 8], output_dim=8)

    butterfly_params = sum(p.numel() for p in butterfly.parameters())
    simple_params = sum(p.numel() for p in simple.parameters())

    assert butterfly_params < simple_params


def test_to_simple_mlp_matches_outputs():
    torch.manual_seed(0)
    model = ButterflyMLP(input_dim=8, hidden_dims=[8], output_dim=8)
    dense = model.to_simple_mlp()

    x = torch.randn(10, 8)
    assert torch.allclose(model(x), dense(x), atol=1e-5, rtol=1e-5)


def test_from_simple_mlp_roundtrip_preserves_outputs():
    torch.manual_seed(42)
    source = ButterflyMLP(input_dim=4, hidden_dims=[4], output_dim=4)
    dense = source.to_simple_mlp()

    reconstructed = ButterflyMLP.from_simple_mlp(dense, seed=0)

    x = torch.randn(12, 4)
    source_out = source(x)
    reconstructed_out = reconstructed(x)

    assert torch.allclose(reconstructed_out, source_out, atol=5e-3, rtol=5e-3)


def test_backward_pass_produces_gradients():
    model = ButterflyMLP(input_dim=4, hidden_dims=[4], output_dim=4)
    x = torch.randn(5, 4)
    target = torch.randn(5, 4)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    gradients = [p.grad for p in model.parameters()]

    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients)
