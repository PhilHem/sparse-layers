import math

import pytest
import torch

from sparse_layers import CustomLinear


def test_initialization_sets_expected_shapes():
    layer = CustomLinear(in_features=5, out_features=3)

    assert layer.weight.shape == (3, 5)
    assert layer.bias is not None
    assert layer.bias.shape == (3,)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_forward_produces_correct_output_shape(batch_size):
    layer = CustomLinear(in_features=6, out_features=2)
    x = torch.randn(batch_size, 6)

    output = layer(x)

    assert output.shape == (batch_size, 2)


def test_matches_nn_linear_when_parameters_are_copied():
    torch.manual_seed(0)
    reference = torch.nn.Linear(7, 4)
    layer = CustomLinear(in_features=7, out_features=4)

    with torch.no_grad():
        layer.weight.copy_(reference.weight)
        if reference.bias is not None:
            layer.bias.copy_(reference.bias)

    x = torch.randn(3, 7)

    expected = reference(x)
    actual = layer(x)

    torch.testing.assert_close(actual, expected)


def test_parameters_are_trainable():
    layer = CustomLinear(in_features=8, out_features=5)

    assert all(param.requires_grad for param in layer.parameters())


def test_initialization_matches_kaiming_uniform():
    torch.manual_seed(1337)
    layer = CustomLinear(in_features=11, out_features=13)

    torch.manual_seed(1337)
    expected_weight = torch.empty_like(layer.weight)
    torch.nn.init.kaiming_uniform_(expected_weight, a=math.sqrt(5))

    torch.testing.assert_close(layer.weight.detach(), expected_weight)

    if layer.bias is not None:
        fan_in = layer.weight.size(1)
        expected_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        expected_bias = torch.empty_like(layer.bias)
        torch.nn.init.uniform_(expected_bias, -expected_bound, expected_bound)

        torch.testing.assert_close(layer.bias.detach(), expected_bias)


def test_requires_positive_dimensions():
    with pytest.raises(ValueError, match="in_features must be a positive integer"):
        CustomLinear(in_features=0, out_features=4)

    with pytest.raises(ValueError, match="out_features must be a positive integer"):
        CustomLinear(in_features=4, out_features=0)


def test_supports_biasless_configuration_and_extra_repr():
    layer = CustomLinear(in_features=3, out_features=2, bias=False)

    assert layer.bias is None
    assert "bias=False" in layer.extra_repr()

