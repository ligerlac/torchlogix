"""LUT parametrization strategies for logic gate networks.
This module provides different ways to parametrize Look-Up Tables (LUTs)
in logic gate networks, using either raw truth table logits,
Walsh-Hadamard basis coefficients, or the indicator-polynomial ('light') basis.
"""

from abc import ABC, abstractmethod
import math
import torch
import torch.nn.functional as F

from .functional import (
    compute_all_logic_ops_vectorized,
    walsh_basis_hard,
    walsh_hadamard_transform,
    light_basis_hard,
    weighted_raw_basis_sum,
    weighted_walsh_basis_sum,
    weighted_light_basis_sum,
    softmax,
    sigmoid,
    gumbel_sigmoid
)


def setup_parametrization(parametrization: str, lut_rank: int, **parametrization_kwargs):
    param_dict = {
        "raw": RawLUTParametrization,
        "warp": WarpLUTParametrization,
        "light": LightLUTParametrization
    }
    if parametrization not in param_dict:
        raise ValueError(
            f"Unsupported parametrization: {parametrization}. "
            f"Choose from {list(param_dict.keys())}."
        )
    param_cls = param_dict[parametrization]
    return param_cls(lut_rank, **parametrization_kwargs)


class LUTParametrization(ABC):
    """Base class for LUT parametrization strategies.

    A parametrization defines how to represent Boolean functions (LUTs)
    as learnable parameters, how to initialize them, and how to compute
    outputs from inputs.
    """

    def __init__(
        self, 
        lut_rank: int, 
        forward_sampling: str = "soft",
        temperature: float = 1.0,
        weight_init: str = "residual",
        residual_probability: float = 0.951,
        materialize_basis: bool = False
    ):
        """Initialize parametrization.

        Args:
            lut_rank: Number of inputs per logic gate (arity of the LUT).
            forward_sampling: Sampling strategy during forward pass. One of:
                - "soft": Continuous relaxation via softmax/sigmoid
                - "hard": Straight-through hard selection
                - "gumbel_soft": Gumbel-Softmax/Sigmoid continuous relaxation
                - "gumbel_hard": Gumbel-Softmax/Sigmoid straight-through
            temperature: Temperature parameter for sampling operations.
            weight_init: Initialization strategy for weights ("residual" or "random").
            residual_probability: Probability controlling residual initialization.
            materialize_basis: Whether to materialize the 2^lut_rank-dimensional basis
                and calculate the sum via the scalar product with weights (strongly discouraged).
                If False, the weighted sum is calculated in-place, saving memory.
        """
        self.lut_rank = lut_rank
        self.lut_entries = 1 << lut_rank

        # Validate and store sampling configuration
        valid_modes = ["soft", "hard", "gumbel_soft", "gumbel_hard"]
        if forward_sampling not in valid_modes:
            raise ValueError(
                f"forward_sampling must be one of {valid_modes}, got {forward_sampling}"
            )
        self.forward_sampling = forward_sampling
        self.temperature = temperature
        self.weight_init = weight_init
        self.residual_probability = residual_probability
        self.materialize_basis = materialize_basis

    @abstractmethod
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        """Initialize weights for this parametrization.

        Args:
            num_neurons: Number of neurons/kernels.
            weight_init: Initialization strategy ("residual" or "random").
            residual_probability: Probability controlling residual initialization.
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

    def update_temperature(self, temperature: float):
        self.temperature = temperature
    

class RawLUTParametrization(LUTParametrization):
    """Raw LUT parametrization using direct truth table logits.

    This parametrization directly assigns logits to each possible truth table
    entry, then samples via softmax. Each LUT of rank n has 2^(2^n) possible
    Boolean functions.
    """

    def __init__(
        self,
        lut_rank: int,
        forward_sampling: str = "soft",
        temperature: float = 1.0,
        weight_init: str = "residual",
        residual_probability: float = 0.951,
        materialize_basis: bool = False
    ):
        super().__init__(
            lut_rank,
            forward_sampling,
            temperature,
            weight_init,
            residual_probability,
            materialize_basis
        )
        if lut_rank != 2:
            raise ValueError("Raw parametrization currently only supports lut_rank=2")
        # Number of possible Boolean functions (not just truth table entries)
        self.num_functions = 1 << self.lut_entries

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            # all weights to 0 except for weight number (1 << (1 << (self.lut_rank - 1))) - 1, 
            # which is set to
            value = math.log((2**lut_entries - 1) * self.residual_probability 
                             - 2**(lut_entries - 1) + 1) - math.log(1 - self.residual_probability)
            weights = torch.zeros((num_neurons, 1 << lut_entries), device=device)
            weights[:, (1 << (1 << (self.lut_rank - 1))) - 1] = value * self.temperature
            return weights
        elif self.weight_init == "random":
            return torch.randn(num_neurons, 1 << lut_entries, device=device)
        raise ValueError(f"Unknown weight_init: {self.weight_init}")


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
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)

        # Extract inputs
        a, b = x[:, 0], x[:, 1]

        # Sample weights (merged from SoftmaxSampler)
        if training:
            w = self._sample_train(weight)
        else:
            w = self._sample_eval(weight)

        # Handle 1D weight case (single neuron) for backward compatibility
        if w.ndim == 1:
            w = w.unsqueeze(0)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            ops = compute_all_logic_ops_vectorized(a, b)  # Shape: (..., 16)
            return torch.einsum(contraction_with_basis_dim, w, ops)

        return weighted_raw_basis_sum(a, b, w, contraction)
    

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


class WarpLUTParametrization(LUTParametrization):
    """Walsh-Hadamard parametrization using basis coefficients.

    This parametrization represents Boolean functions via Walsh-Hadamard
    basis coefficients, which provides a smoother optimization landscape
    and exploits structure in Boolean functions (parity, correlations).
    """

    def __init__(
        self,
        lut_rank: int,
        forward_sampling: str = "soft",
        temperature: float = 1.0,
        weight_init: str = "residual",
        residual_probability: float = 0.951,
        materialize_basis: bool = False
    ):
        super().__init__(
            lut_rank,
            forward_sampling,
            temperature,
            weight_init,
            residual_probability,
            materialize_basis
        )
        if lut_rank not in [1, 2, 4, 6]:
            raise ValueError(
                f"Hard-coded Walsh basis only supports lut_rank in [1, 2, 4, 6], got {lut_rank}"
            )

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str,
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            weights = torch.empty((num_neurons, lut_entries), device=device)
            # identity representation, corresponds to Boolean function, which maps MSB (last single variable) to itself
            identity = 1 - 2 * torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
            transformed_identity = (1 / lut_entries) * walsh_hadamard_transform(identity, self.lut_rank, dtype=torch.int32, device=device)
            # transformation for comparibility with raw parametrization
            c_walsh = self.temperature * (math.log(self.residual_probability) - math.log(1 - self.residual_probability))
            weights[:] = c_walsh * transformed_identity.to(torch.float)
            # add random noise proportional to temperature (breaks the tests but helps optimization by breaking symmetry between neurons)
            # weights += self.temperature * 0.1 * torch.randn(num_neurons, lut_entries, device=device)
            return weights
        elif self.weight_init == "residual-catalog":
            identity_truth_table = 1 - 2 * torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
            identity_coefficients = (1 / lut_entries) * walsh_hadamard_transform(identity_truth_table, self.lut_rank, dtype=torch.int32, device=device)
            
            # build weights tensor such that all gates are identity initially
            weights = torch.empty((num_neurons, lut_entries), device=device)
            weights[:] = identity_coefficients

            # set 1 - residual_probability fraction of weights to random different functions
            n_different = round(num_neurons * (1 - self.residual_probability))
            random_truth_tables = torch.randint(0, 2, (n_different, lut_entries), device=device)
            random_truth_tables = 1 - 2 * random_truth_tables.to(dtype=torch.int32)
            random_coefficients = (1 / lut_entries) * walsh_hadamard_transform(random_truth_tables, self.lut_rank, dtype=torch.int32, device=device)

            # randomly pick indices to replace
            indices = torch.randperm(num_neurons, device=device)
            weights[indices[:n_different]] = random_coefficients
            return weights
        elif self.weight_init == "random":
            return torch.randn(num_neurons, lut_entries, device=device)
        else:
            raise ValueError(self.weight_init)

    
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

        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)

        # Convert to [-1, +1]
        x = 1 - 2 * x

        if weight.ndim == 1:
            weight = weight.unsqueeze(0)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            x = walsh_basis_hard(x, self.lut_rank)
            x = torch.einsum(contraction_with_basis_dim, weight, x)
        
        else:
            x = weighted_walsh_basis_sum(x, weight, contraction, self.lut_rank)

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

    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        luts = walsh_hadamard_transform(weight, self.lut_rank)
        return luts < 0

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lut_rank <= 4, "LUT IDs only supported for lut_rank <= 4 due to combinatorial explosion."
        luts = self.get_luts(weight)

        ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
        ids = (luts * ids.unsqueeze(0)).sum(dim=1)

        return luts, ids
    

class LightLUTParametrization(LUTParametrization):
    """Light DGN parametrization using basis coefficients.

    This parametrization represents Boolean functions via positive
    basis coefficients, which are mapped to [0,1] with sigmoid.
    """

    def __init__(
        self,
        lut_rank: int,
        forward_sampling: str = "soft",
        temperature: float = 1.0,
        weight_init: str = "residual",
        residual_probability: float = 0.951,
        materialize_basis: bool = False
    ):
        super().__init__(
            lut_rank,
            forward_sampling,
            temperature,
            weight_init,
            residual_probability,
            materialize_basis
        )
        if lut_rank not in [2, 4, 6]:
            raise ValueError(
                f"Hard-coded Light basis only supports lut_rank in [2, 4, 6], got {lut_rank}"
            )

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            weights = torch.randn(num_neurons, lut_entries, device=device)
            weights[:, :lut_entries // 2] -= 3
            weights[:, lut_entries // 2:] += 3
            return weights
        elif self.weight_init == "random":
            return torch.rand(num_neurons, lut_entries, device=device)
        else:
            raise ValueError(self.weight_init)
    

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
        # Handle 1D weight case (single neuron) for backward compatibility
        if weight.ndim == 1:
            weight = weight.unsqueeze(0)

        if training:
            w = self._sample_train(weight)
        else:
            w = self._sample_eval(weight)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            x = light_basis_hard(x, self.lut_rank)
            x = torch.einsum(contraction_with_basis_dim, w, x)
        else:
            x = weighted_light_basis_sum(x, w, contraction, self.lut_rank)
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

    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        luts = weight > 0
        return luts

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lut_rank <= 4, "LUT IDs only supported for lut_rank <= 4 due to combinatorial explosion."
        luts = self.get_luts(weight)

        ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
        ids = (luts * ids.unsqueeze(0)).sum(dim=1)

        return luts, ids
