import pytest
import torch

from sparse_layers import CustomLinear, CustomMLP, SimpleMLP


def _extract_custom_linear_layers(model: CustomMLP) -> list[CustomLinear]:
    return [module for module in model.modules() if isinstance(module, CustomLinear)]


def test_initialization_creates_expected_layers():
    model = CustomMLP(input_dim=10, hidden_dims=[16, 8], output_dim=4)

    linear_layers = _extract_custom_linear_layers(model)

    assert len(linear_layers) == 3
    assert linear_layers[0].in_features == 10 and linear_layers[0].out_features == 16
    assert linear_layers[1].in_features == 16 and linear_layers[1].out_features == 8
    assert linear_layers[2].in_features == 8 and linear_layers[2].out_features == 4


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_forward_pass_has_correct_shape(batch_size):
    model = CustomMLP(input_dim=6, hidden_dims=[12], output_dim=3)
    x = torch.randn(batch_size, 6)

    output = model(x)

    assert output.shape == (batch_size, 3)


def test_parameters_are_trainable():
    model = CustomMLP(input_dim=5, hidden_dims=[10], output_dim=2)

    assert all(param.requires_grad for param in model.parameters())


def test_forward_requires_2d_input():
    model = CustomMLP(input_dim=5, hidden_dims=[10], output_dim=2)
    x = torch.randn(5)

    with pytest.raises(ValueError):
        model(x)


@pytest.mark.parametrize(
    "input_dim, hidden_dims, output_dim, expected_message",
    [
        (0, [8], 4, "input_dim must be a positive integer"),
        (5, [], 4, "hidden_dims must contain at least one positive integer"),
        (5, [8, 0], 4, "hidden_dims must contain only positive integers"),
        (5, [8], -1, "output_dim must be a positive integer"),
    ],
)
def test_invalid_configuration_raises_value_error(
    input_dim, hidden_dims, output_dim, expected_message
):
    with pytest.raises(ValueError, match=expected_message):
        CustomMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)


def test_matches_simple_mlp_when_weights_are_copied():
    torch.manual_seed(123)
    simple = SimpleMLP(input_dim=7, hidden_dims=[5], output_dim=3)

    torch.manual_seed(123)
    custom = CustomMLP(input_dim=7, hidden_dims=[5], output_dim=3)

    simple_linear_layers = [
        module for module in simple.modules() if isinstance(module, torch.nn.Linear)
    ]
    custom_linear_layers = _extract_custom_linear_layers(custom)

    with torch.no_grad():
        for custom_layer, simple_layer in zip(custom_linear_layers, simple_linear_layers):
            custom_layer.weight.copy_(simple_layer.weight)
            if custom_layer.bias is not None and simple_layer.bias is not None:
                custom_layer.bias.copy_(simple_layer.bias)

    x = torch.randn(4, 7)

    torch.testing.assert_close(custom(x), simple(x))

