import pytest
import torch

from sparse_layers import PaddedButterflyLinear


@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (6, 6),  # non-power-of-two square
        (5, 3),  # non-square
        (8, 8),  # already power-of-two
    ],
)
def test_padded_butterfly_linear_preserves_output_shape(in_features, out_features):
    layer = PaddedButterflyLinear(in_features, out_features)
    x = torch.randn(4, in_features)

    output = layer(x)

    assert output.shape == (4, out_features)


@pytest.mark.parametrize("in_features, out_features", [(0, 4), (-3, 2), (4, 0)])
def test_padded_butterfly_linear_rejects_invalid_dimensions(in_features, out_features):
    with pytest.raises(ValueError):
        PaddedButterflyLinear(in_features, out_features)


def test_padded_butterfly_linear_parameter_count_reduced_vs_dense():
    in_features = 30
    out_features = 30

    layer = PaddedButterflyLinear(in_features, out_features, bias=False)
    dense = torch.nn.Linear(in_features, out_features, bias=False)

    butterfly_params = sum(p.numel() for p in layer.parameters())
    dense_params = sum(p.numel() for p in dense.parameters())

    assert butterfly_params < dense_params


def test_padded_butterfly_linear_matches_butterfly_for_power_of_two():
    torch.manual_seed(0)
    in_features = out_features = 8
    x = torch.randn(3, in_features)

    layer = PaddedButterflyLinear(in_features, out_features, bias=True)
    direct = PaddedButterflyLinear(in_features, out_features, bias=True)
    direct.load_state_dict(layer.state_dict())

    assert torch.allclose(layer(x), direct(x))


def test_padded_butterfly_linear_backward_produces_gradients():
    torch.manual_seed(1)
    layer = PaddedButterflyLinear(6, 6, bias=True)
    x = torch.randn(5, 6, requires_grad=True)
    target = torch.randn(5, 6)

    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    gradients = [p.grad for p in layer.parameters() if p.requires_grad]

    assert gradients, "Expected at least one learnable parameter"
    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert torch.isfinite(x.grad).all()
