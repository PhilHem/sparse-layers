from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from sparse_layers.models._baselines import SimpleMLP
from sparse_layers.modules.butterfly_linear import ButterflyLinear
from sparse_layers.ops.butterfly import _is_power_of_two


class ButterflyMLP(nn.Module):
    """A multi-layer perceptron composed of :class:`ButterflyLinear` layers."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()

        self._validate_dimensions(input_dim, hidden_dims, output_dim)

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        layer_dims = [input_dim, *self.hidden_dims, output_dim]

        modules: list[nn.Module] = []
        for idx in range(len(layer_dims) - 1):
            in_features = layer_dims[idx]
            out_features = layer_dims[idx + 1]

            linear = ButterflyLinear(in_features, out_features)
            modules.append(linear)

            if idx < len(layer_dims) - 2:
                modules.append(nn.ReLU())

        self.network = nn.Sequential(*modules)

    @staticmethod
    def _validate_dimensions(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one positive integer")

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers")

        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")

        all_dims = [input_dim, *hidden_dims, output_dim]
        if len(set(all_dims)) != 1:
            raise ValueError("ButterflyMLP requires all layer dimensions to be identical")

        if not all(_is_power_of_two(dim) for dim in all_dims):
            raise ValueError("ButterflyMLP requires power-of-two layer dimensions")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("ButterflyMLP expects a 2D input tensor of shape (batch, features)")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, received {x.shape[1]}"
            )

        return self.network(x)

    def to_simple_mlp(self) -> SimpleMLP:
        """Return a :class:`SimpleMLP` with identical behaviour."""

        simple = SimpleMLP(self.input_dim, self.hidden_dims, self.output_dim)

        sparse_layers = [module for module in self.network if isinstance(module, ButterflyLinear)]
        dense_layers = [module for module in simple.modules() if isinstance(module, nn.Linear)]

        if len(sparse_layers) != len(dense_layers):
            raise RuntimeError("Unexpected layer mismatch during conversion to SimpleMLP")

        for butterfly_layer, dense_layer in zip(sparse_layers, dense_layers, strict=False):
            converted = butterfly_layer.to_linear()
            with torch.no_grad():
                dense_layer.weight.copy_(converted.weight)
                if dense_layer.bias is not None and converted.bias is not None:
                    dense_layer.bias.copy_(converted.bias)

        return simple

    @classmethod
    def from_simple_mlp(
        cls,
        model: SimpleMLP,
        *,
        seed: int | None = None,
        optimization_steps: int = 4000,
        learning_rate: float = 0.1,
        tolerance: float = 1e-7,
    ) -> ButterflyMLP:
        """Construct a :class:`ButterflyMLP` from a compatible :class:`SimpleMLP`."""

        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

        if not linear_layers:
            raise ValueError("SimpleMLP must contain at least one linear layer")

        layer_dims = []
        for layer in linear_layers:
            if layer.in_features != layer.out_features:
                raise ValueError("ButterflyMLP.from_simple_mlp requires square nn.Linear layers")
            if not _is_power_of_two(layer.in_features):
                raise ValueError("ButterflyMLP.from_simple_mlp requires power-of-two dimensions")
            layer_dims.append(layer.in_features)

        input_dim = linear_layers[0].in_features
        hidden_dims = [layer.out_features for layer in linear_layers[:-1]]
        output_dim = linear_layers[-1].out_features

        result = cls(input_dim, hidden_dims, output_dim)

        butterfly_indices = [
            idx for idx, module in enumerate(result.network) if isinstance(module, ButterflyLinear)
        ]

        if len(butterfly_indices) != len(linear_layers):
            raise RuntimeError("Unexpected layer mismatch during reconstruction from SimpleMLP")

        for _index, (linear_layer, target_idx) in enumerate(
            zip(linear_layers, butterfly_indices, strict=False)
        ):
            converted = ButterflyLinear.from_linear(
                linear_layer,
                seed=seed,
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                tolerance=tolerance,
            )
            result.network[target_idx] = converted

        return result
