from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sparse_layers.ops.butterfly import _is_power_of_two


class ButterflyLinear(nn.Module):
    """Butterfly factorization based linear layer.

    This layer follows the structure described in section 2.3.1 of the
    `reducing_memory_requirements_ipu_butterfly.md` reference. It consists of
    log2(N) stages of block-diagonal 2x2 butterfly factors, each stored as a
    learnable parameter. The layer currently supports square power-of-two
    dimensions and acts as a drop-in replacement for :class:`torch.nn.Linear`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        if in_features != out_features:
            raise ValueError("ButterflyLinear requires in_features == out_features")

        if not _is_power_of_two(in_features):
            raise ValueError(
                "ButterflyLinear requires dimensions that are a power of two"
            )

        self.in_features = in_features
        self.out_features = out_features
        self._depth = int(math.log2(in_features))

        num_blocks = in_features // 2
        default_dtype = torch.get_default_dtype()
        identity_block = torch.eye(2, dtype=default_dtype).unsqueeze(0)

        self.factors = nn.ParameterList()
        for _ in range(self._depth):
            factor = identity_block.repeat(num_blocks, 1, 1)
            factor += 0.01 * torch.randn_like(factor)
            self.factors.append(nn.Parameter(factor))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=default_dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        optimization_steps: int = 4000,
        learning_rate: float = 0.1,
        tolerance: float = 1e-7,
        seed: int | None = None,
    ) -> "ButterflyLinear":
        """Construct a butterfly layer approximating a dense :class:`nn.Linear`.

        The method uses gradient-based fitting over the canonical basis to match
        the transformation represented by ``layer``. The bias term is copied
        directly before optimisation to accelerate convergence.
        """

        if layer.in_features != layer.out_features:
            raise ValueError("ButterflyLinear.from_linear requires a square nn.Linear")

        if not _is_power_of_two(layer.in_features):
            raise ValueError(
                "ButterflyLinear.from_linear requires dimensions that are a power of two"
            )

        device = layer.weight.device
        dtype = layer.weight.dtype

        if seed is not None:
            torch.manual_seed(seed)

        result = cls(layer.in_features, layer.out_features, bias=layer.bias is not None)
        result = result.to(device=device, dtype=dtype)

        if result.bias is not None and layer.bias is not None:
            with torch.no_grad():
                result.bias.copy_(layer.bias)

        params = list(result.factors.parameters())
        if not params:
            return result

        optimizer_adam = torch.optim.Adam(params, lr=learning_rate)

        eye_input = torch.eye(layer.in_features, device=device, dtype=dtype)
        with torch.no_grad():
            target = layer(eye_input)

        best_loss = float("inf")

        for step in range(optimization_steps):
            optimizer_adam.zero_grad()
            output = result(eye_input)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer_adam.step()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss

            if best_loss <= tolerance:
                break

        if best_loss > tolerance:
            optimizer_lbfgs = torch.optim.LBFGS(
                params,
                lr=1.0,
                max_iter=100,
                tolerance_grad=1e-12,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                optimizer_lbfgs.zero_grad()
                output_lbfgs = result(eye_input)
                lbfgs_loss = F.mse_loss(output_lbfgs, target)
                lbfgs_loss.backward()
                return lbfgs_loss

            for _ in range(20):
                loss = optimizer_lbfgs.step(closure)
                best_loss = min(best_loss, loss.item())
                if best_loss <= tolerance:
                    break

        return result

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {input.shape[-1]}"
            )

        original_shape = input.shape[:-1]
        x = input.reshape(-1, self.in_features)

        for stage_index, factor in enumerate(self.factors):
            x = self._apply_stage(x, factor, stage_index)

        if self.bias is not None:
            x = x + self.bias

        return x.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def _apply_stage(self, x: Tensor, factor: Tensor, stage: int) -> Tensor:
        batch = x.shape[0]
        n = self.in_features
        block = 1 << (stage + 1)
        half = block >> 1

        staged = x.reshape(batch, -1, block)
        staged = staged.reshape(batch, -1, half, 2)
        staged = staged.permute(0, 1, 3, 2).contiguous()
        pairs = staged.reshape(batch, -1, 2)

        transformed = torch.einsum("bnc,ncd->bnd", pairs, factor)

        transformed = transformed.reshape(batch, -1, 2, half)
        transformed = transformed.permute(0, 1, 3, 2).contiguous()
        transformed = transformed.reshape(batch, -1, block)
        return transformed.reshape(batch, n)

    def to_linear(self) -> nn.Linear:
        """Return a dense :class:`nn.Linear` with identical behaviour."""

        factor_tensor = self.factors[0] if len(self.factors) > 0 else None
        if factor_tensor is not None:
            device = factor_tensor.device
            dtype = factor_tensor.dtype
        elif self.bias is not None:
            device = self.bias.device
            dtype = self.bias.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            if self.bias is not None:
                bias_backup = self.bias.data.clone()
                self.bias.zero_()
            else:
                bias_backup = None

            identity = torch.eye(self.in_features, device=device, dtype=dtype)
            weight_matrix = self(identity)

            if bias_backup is not None:
                self.bias.copy_(bias_backup)
                linear.bias.copy_(bias_backup)

            linear.weight.copy_(weight_matrix.t())

        return linear
