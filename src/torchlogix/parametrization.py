"""LUT parametrization strategies for logic gate networks.

This module provides different ways to parametrize Look-Up Tables (LUTs)
in logic gate networks, using either raw truth table logits or Walsh-Hadamard
basis coefficients.
"""

from abc import ABC, abstractmethod
import math
import torch
import torch.nn.functional as F

from .functional import (
    walsh_basis_hard,
    walsh_hadamard_transform,
    compute_all_logic_ops_vectorized,
    softmax,
    sigmoid,
    gumbel_sigmoid
)


class LUTParametrization(ABC):
    """Base class for LUT parametrization strategies.

    A parametrization defines how to represent Boolean functions (LUTs)
    as learnable parameters, how to initialize them, and how to compute
    outputs from inputs.
    """

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False,
                 forward_sampling: str = "soft", temperature: float = 1.0):
        """Initialize parametrization.

        Args:
            lut_rank: Number of inputs per logic gate (arity of the LUT).
            arbitrary_basis: If True, allows arbitrary basis functions
                rather than hard-coded optimized implementations.
            forward_sampling: Sampling strategy during forward pass. One of:
                - "soft": Continuous relaxation via softmax/sigmoid
                - "hard": Straight-through hard selection
                - "gumbel_soft": Gumbel-Softmax/Sigmoid continuous relaxation
                - "gumbel_hard": Gumbel-Softmax/Sigmoid straight-through
            temperature: Temperature parameter for sampling operations.
        """
        self.lut_rank = lut_rank
        self.lut_entries = 1 << lut_rank
        self.arbitrary_basis = arbitrary_basis

        # Validate and store sampling configuration
        valid_modes = ["soft", "hard", "gumbel_soft", "gumbel_hard"]
        if forward_sampling not in valid_modes:
            raise ValueError(
                f"forward_sampling must be one of {valid_modes}, got {forward_sampling}"
            )
        self.forward_sampling = forward_sampling
        self.temperature = temperature

    @abstractmethod
    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_probability: float,
        device: str
    ) -> torch.Tensor:
        """Initialize weights for this parametrization.

        Args:
            num_neurons: Number of neurons/kernels.
            weight_init: Initialization strategy ("residual" or "random").
            residual_probability: Parameter controlling residual initialization.
            device: Device to allocate weights on.

        Returns:
            Initialized weight tensor.
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        training: bool,
        contraction: str
    ) -> torch.Tensor:
        """Compute forward pass (layer-agnostic).

        Args:
            x: Extracted inputs with lut_rank at dimension 1, shape (batch, lut_rank, ...)
            weight: Weight parameters
            training: Whether in training mode
            contraction: Einsum pattern specifying how to combine basis/ops with weights
                        (e.g., 'bnk,nk->bn' for dense, 'bcsfk,fck->bcsf' for conv)

        Returns:
            Output tensor with lut_rank dimension reduced
        """
        pass

    @abstractmethod
    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        """Extract LUT truth tables from weights.

        Args:
            weight: Weight parameters.

        Returns:
                - luts: Boolean tensor of truth tables
        """
        pass

    @abstractmethod
    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False,
                 forward_sampling: str = "soft", temperature: float = 1.0):
        super().__init__(lut_rank, arbitrary_basis, forward_sampling, temperature)
        if lut_rank != 2:
            raise ValueError("Raw parametrization currently only supports lut_rank=2")
        # Number of possible Boolean functions (not just truth table entries)
        self.num_functions = 1 << self.lut_entries

    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_probability: float,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if weight_init == "residual":
            # all weights to 0 except for weight number lut_entries - 1, 
            # which is set to residual_probability corresponding to a probability
            # If we want to sample the identity function with probability p, the param is:
            if lut_entries < 5:
                param = - math.log(1 - residual_probability) + math.log(residual_probability) + math.log((1 << lut_entries) - 1)
            else:
                param = - math.log(1 - residual_probability) + math.log(residual_probability) + lut_entries * math.log(2)
            # Original DGN paper set this parameter to 5, which leads to a probability of ~0.9082 for lut_rank=2
            weights = torch.zeros((num_neurons, 1 << lut_entries), device=device)
            weights[:, lut_entries - 1] = param
            return weights
        elif weight_init == "random":
            return torch.randn(num_neurons, 1 << lut_entries, device=device)
        raise ValueError(f"Unknown weight_init: {weight_init}")

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        training: bool,
        contraction: str
    ) -> torch.Tensor:
        """Layer-agnostic forward pass.

        Args:
            x: Extracted inputs, shape (batch, lut_rank, ...)
            weight: Weight parameters
            training: Whether in training mode
            contraction: Einsum pattern for combining ops with weights

        Returns:
            Output with lut_rank dimension reduced
        """
        # Extract inputs
        a, b = x[:, 0], x[:, 1]

        # Sample weights (merged from SoftmaxSampler)
        if training:
            w = self._sample_train(weight)
        else:
            w = self._sample_eval(weight)

        # Compute all logic operations
        ops = compute_all_logic_ops_vectorized(a, b)  # Shape: (..., 16)

        # Apply layer-provided contraction (no layer-type detection)
        # Convert ops to same dtype as weights for einsum compatibility
        ops = ops.to(w.dtype)

        # Handle 1D weight case (single neuron) for backward compatibility
        if w.ndim == 1:
            w = w.unsqueeze(0)

        return torch.einsum(contraction, ops, w)

    def _sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply softmax/gumbel_softmax based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return softmax(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return softmax(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return F.gumbel_softmax(weights, tau=self.temperature, hard=False, dim=-1)
        elif self.forward_sampling == "gumbel_hard":
            return F.gumbel_softmax(weights, tau=self.temperature, hard=True, dim=-1)

    def _sample_eval(self, weights: torch.Tensor) -> torch.Tensor:
        """Select most probable LUT via argmax + one-hot."""
        return F.one_hot(
            weights.argmax(-1), self.num_functions
        ).to(torch.float32)

    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        ids = weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(self.lut_entries, device=ids.device)) & 1).flip(1)
        return luts

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(self.lut_entries, device=ids.device)) & 1).flip(1)
        return luts, ids


class WalshLUTParametrization(LUTParametrization):
    """Walsh-Hadamard parametrization using basis coefficients.

    This parametrization represents Boolean functions via Walsh-Hadamard
    basis coefficients, which provides a smoother optimization landscape
    and exploits structure in Boolean functions (parity, correlations).
    """

    def __init__(self, lut_rank: int, arbitrary_basis: bool = False,
                 forward_sampling: str = "soft", temperature: float = 1.0):
        super().__init__(lut_rank, arbitrary_basis, forward_sampling, temperature)
        if not arbitrary_basis and lut_rank not in [2, 4, 6]:
            raise ValueError(
                f"Hard-coded Walsh basis only supports lut_rank in [2, 4, 6], got {lut_rank}"
            )

    def init_weights(
        self,
        num_neurons: int,
        weight_init: str,
        residual_probability: float,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if weight_init == "residual":
            n = int((num_neurons * residual_probability)) % (num_neurons + 1)
            weights = torch.empty((num_neurons, lut_entries), device=device)
            # identity representation, corresponds to Boolean function, which maps MSB (last single variable) to itself
            identity = torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
            transformed_identity = walsh_hadamard_transform(identity, self.lut_rank, dtype=torch.int32, device=device)
            # randomly sample indices
            indices = torch.randperm(num_neurons, device=device)
            weights[indices[:n]] = (1.0 / self.lut_rank) * transformed_identity.to(torch.float)
            # sample random binary representations
            samples = 1 - 2 * torch.randint(0, 2, (num_neurons - n, lut_entries), device=device).to(torch.int32)
            # convert with wh transform
            transformed_samples = walsh_hadamard_transform(samples, self.lut_rank, device=device, dtype=torch.int32)
            if n < num_neurons:
                weights[indices[n:]] = (1.0 / self.lut_rank) * transformed_samples.to(torch.float)
            return weights
        elif weight_init == "random":
            return torch.randn(num_neurons, lut_entries, device=device) * 0.1
        else:
            raise ValueError(weight_init)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        training: bool,
        contraction: str
    ) -> torch.Tensor:
        """Layer-agnostic forward pass.

        Args:
            x: Extracted inputs, shape (batch, lut_rank, ...)
            weight: Weight parameters
            training: Whether in training mode
            contraction: Einsum pattern for combining basis with weights

        Returns:
            Output with lut_rank dimension reduced
        """

        # Convert to [-1, +1]
        x = 1 - 2 * x

        # Compute Walsh basis
        if not self.arbitrary_basis:
            basis = walsh_basis_hard(x, self.lut_rank)
        else:
            raise NotImplementedError("Arbitrary basis requires layer context")

        # Handle 1D weight case (single neuron) for backward compatibility
        if weight.ndim == 1:
            weight = weight.unsqueeze(0)

        # Apply layer-provided contraction (no layer-type detection)
        # Convert basis to same dtype as weights for einsum compatibility
        x = torch.einsum(contraction, basis.to(weight.dtype), weight)

        # Sample output (merged from SigmoidSampler)
        if training:
            x = self._sample_train(-x)
        else:
            x = self._sample_eval(-x)

        return x

    def _sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid/gumbel_sigmoid based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return sigmoid(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=True)

    def _sample_eval(self, weights: torch.Tensor) -> torch.Tensor:
        """Threshold at 0 for discrete output."""
        return (weights > 0).to(dtype=torch.float32)

    def get_luts(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luts = walsh_hadamard_transform(weight, self.lut_rank)
        return luts < 0

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lut_rank <= 4, "LUT IDs only supported for lut_rank <= 4 due to combinatorial explosion."
        luts = self.get_luts(weight)

        ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
        ids = (luts * ids.unsqueeze(0)).sum(dim=1)

        return luts, ids
