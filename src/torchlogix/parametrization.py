"""LUT parametrization strategies for logic gate networks.

This module provides different ways to parametrize Look-Up Tables (LUTs)
in logic gate networks, using either raw truth table logits or Walsh-Hadamard
basis coefficients.
"""

from abc import ABC, abstractmethod
import torch

from .initialization import initialize_weights_raw, initialize_weights_walsh
from .functional import (
    bin_op_s,
    bin_op_cnn,
    walsh_basis_hard,
    walsh_basis_hard_cnn_deep_level,
    walsh_cnn,
    walsh_hadamard_transform
)


class LUTParametrization(ABC):
    """Base class for LUT parametrization strategies.

    A parametrization defines how to represent Boolean functions (LUTs)
    as learnable parameters, how to initialize them, and how to compute
    outputs from inputs.
    """

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False):
        """Initialize parametrization.

        Args:
            lut_rank: Number of inputs per logic gate (arity of the LUT).
            arbitrary_basis: If True, allows arbitrary basis functions
                rather than hard-coded optimized implementations.
        """
        self.lut_rank = lut_rank
        self.lut_entries = 1 << lut_rank
        self.arbitrary_basis = arbitrary_basis

    @abstractmethod
    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_init_param: float,
        device: str
    ) -> torch.Tensor:
        """Initialize weights for this parametrization.

        Args:
            num_neurons: Number of neurons/kernels.
            weight_init: Initialization strategy ("residual" or "random").
            residual_init_param: Parameter controlling residual initialization.
            device: Device to allocate weights on.

        Returns:
            Initialized weight tensor.
        """
        pass

    @abstractmethod
    def forward_dense(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        """Compute forward pass for dense logic layer.

        Args:
            a: First input tensor of shape (..., out_dim).
            b: Second input tensor of shape (..., out_dim).
            weight: Weight parameters.
            sampler: Sampler instance for weight sampling.
            training: Whether in training mode.

        Returns:
            Output tensor of shape (..., out_dim).
        """
        pass

    @abstractmethod
    def forward_conv_first_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        """Compute forward pass for first level of convolutional logic tree.

        Args:
            x: Input tensor from receptive field extraction.
            weight: Weight parameters for this level.
            sampler: Sampler instance.
            training: Whether in training mode.

        Returns:
            Output tensor from first tree level.
        """
        pass

    @abstractmethod
    def forward_conv_deep_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        """Compute forward pass for deeper levels of convolutional logic tree.

        Args:
            x: Input tensor from previous tree level.
            weight: Weight parameters for this level.
            sampler: Sampler instance.
            training: Whether in training mode.

        Returns:
            Output tensor from this tree level.
        """
        pass

    @abstractmethod
    def get_lut_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract LUT truth tables and IDs from weights.

        Args:
            weight: Weight parameters.

        Returns:
            Tuple of (luts, ids) where:
                - luts: Boolean tensor of truth tables
                - ids: Integer tensor of LUT IDs (or None if not applicable)
        """
        pass


class RawLUTParametrization(LUTParametrization):
    """Raw LUT parametrization using direct truth table logits.

    This parametrization directly assigns logits to each possible truth table
    entry, then samples via softmax. Each LUT of rank n has 2^(2^n) possible
    Boolean functions.
    """

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False):
        super().__init__(lut_rank, arbitrary_basis)
        if lut_rank != 2:
            raise ValueError("Raw parametrization currently only supports lut_rank=2")
        # Number of possible Boolean functions (not just truth table entries)
        self.num_functions = 1 << self.lut_entries

    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_init_param: float,
        device: str
    ) -> torch.Tensor:
        return initialize_weights_raw(
            weight_init, num_neurons, self.lut_rank, residual_init_param, device
        )

    def forward_dense(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        a, b = x[:, 0], x[:, 1]

        if training:
            w = sampler.sample_train(weight)
        else:
            w = sampler.sample_eval(weight, self.num_functions)

        return bin_op_s(a, b, w)

    def forward_conv_first_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        if training:
            level_weights = torch.stack([sampler.sample_train(w) for w in weight], dim=0)
        else:
            level_weights = torch.stack(
                [sampler.sample_eval(w, self.num_functions) for w in weight], dim=0
            )

        return bin_op_cnn(x[:, 0], x[:, 1], level_weights)

    def forward_conv_deep_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        if training:
            level_weights = torch.stack([sampler.sample_train(w) for w in weight], dim=0)
        else:
            level_weights = torch.stack(
                [sampler.sample_eval(w, self.num_functions) for w in weight], dim=0
            )

        return bin_op_cnn(x[..., 0, :], x[..., 1, :], level_weights)

    def get_lut_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(self.lut_entries, device=ids.device)) & 1).flip(1)
        return luts, ids


class WalshLUTParametrization(LUTParametrization):
    """Walsh-Hadamard parametrization using basis coefficients.

    This parametrization represents Boolean functions via Walsh-Hadamard
    basis coefficients, which provides a smoother optimization landscape
    and exploits structure in Boolean functions (parity, correlations).
    """

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False):
        super().__init__(lut_rank, arbitrary_basis)
        if not arbitrary_basis and lut_rank not in [2, 4, 6]:
            raise ValueError(
                f"Hard-coded Walsh basis only supports lut_rank in [2, 4, 6], got {lut_rank}"
            )

    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_init_param: float,
        device: str
    ) -> torch.Tensor:
        return initialize_weights_walsh(
            weight_init, num_neurons, self.lut_rank, residual_init_param, device
        )

    def forward_dense(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        x = 1 - 2 * x  # Convert to {-1, +1}

        if not self.arbitrary_basis:
            basis = walsh_basis_hard(x, self.lut_rank)
        else:
            # Generic basis computation using precomputed bits
            # This would require access to self.bits from the layer
            raise NotImplementedError("Arbitrary basis requires layer context")

        x = (weight * basis).sum(dim=-1)

        if training:
            x = sampler.sample_train(-x)
        else:
            x = sampler.sample_eval(-x)

        return x

    def forward_conv_first_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        level_weights = torch.stack([w for w in weight], dim=0)

        if not self.arbitrary_basis:
            basis = walsh_basis_hard(x, self.lut_rank)
        else:
            raise NotImplementedError("Arbitrary basis not implemented yet")

        x = walsh_cnn(basis, level_weights)

        if training:
            x = sampler.sample_train(-x)
        else:
            x = sampler.sample_eval(-x)

        return x

    def forward_conv_deep_level(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sampler,
        training: bool
    ) -> torch.Tensor:
        level_weights = torch.stack([w for w in weight], dim=0)

        if not self.arbitrary_basis:
            basis = walsh_basis_hard_cnn_deep_level(x, self.lut_rank)
        else:
            raise NotImplementedError("Arbitrary basis not implemented yet")

        x = walsh_cnn(basis, level_weights)

        if training:
            x = sampler.sample_train(-x)
        else:
            x = sampler.sample_eval(-x)

        return x

    def get_lut_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luts = walsh_hadamard_transform(weight, self.lut_rank)
        luts = luts < 0

        if self.lut_rank <= 4:
            ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
            ids = (luts * ids.unsqueeze(0)).sum(dim=1)
        else:
            ids = None

        return luts, ids
