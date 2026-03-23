import pytest
import torch

from butterfly_layers import ButterflyLinear


def test_initialization_sets_in_out_features():
    layer = ButterflyLinear(in_features=4, out_features=4, bias=True)

    assert layer.in_features == 4
    assert layer.out_features == 4


@pytest.mark.parametrize("in_features, out_features", [(0, 4), (4, 0)])
def test_invalid_dimensions_raise_value_error(in_features, out_features):
    with pytest.raises(ValueError):
        ButterflyLinear(in_features=in_features, out_features=out_features)


def test_forward_shape_matches_linear():
    layer = ButterflyLinear(in_features=8, out_features=8, bias=False)
    x = torch.randn(4, 8)

    output = layer(x)

    assert output.shape == (4, 8)


def test_requires_power_of_two_dimensions():
    with pytest.raises(ValueError, match="power of two"):
        ButterflyLinear(in_features=6, out_features=6)


def test_parameter_count_reduced_compared_to_dense():
    layer = ButterflyLinear(in_features=8, out_features=8, bias=False)
    dense = torch.nn.Linear(8, 8, bias=False)

    butterfly_params = sum(p.numel() for p in layer.parameters())
    dense_params = sum(p.numel() for p in dense.parameters())

    assert butterfly_params < dense_params


def test_from_linear_matches_dense_output():
    torch.manual_seed(0)
    source = ButterflyLinear(in_features=4, out_features=4, bias=True)
    dense = source.to_linear()
    layer = ButterflyLinear.from_linear(dense, seed=0)

    x = torch.randn(16, 4)
    source_out = source(x)
    dense_out = dense(x)
    butterfly_out = layer(x)

    assert torch.allclose(butterfly_out, dense_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(butterfly_out, source_out, atol=1e-4, rtol=1e-4)


def test_backward_pass_produces_gradients():
    layer = ButterflyLinear(in_features=8, out_features=8, bias=True)
    x = torch.randn(5, 8)
    target = torch.randn(5, 8)

    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    gradients = [p.grad for p in layer.parameters()]

    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients)
