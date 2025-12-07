import torch

from torchlogix.layers.connections import setup_connections

from ..functional import (
    GradFactor, get_combination_indices, get_regularization_loss, rescale_weights, take_tuples
    )
from .base import LogicBase


class LogicDense(LogicBase):
    """Fully-connected logic gate layer with differentiable learning.

    This module provides the core implementation of Differentiable Deep Logic
    Gate Networks. Each neuron learns a Boolean logic function (LUT) that
    operates on a subset of input features.

    Args:
        in_dim: Number of input features (last dimension of the input tensor).
        out_dim: Number of output neurons (logical units).
        parametrization: LUT parametrization method. One of:
            - ``"raw"``: Direct truth table logits (supports lut_rank=2).
            - ``"walsh"``: Walsh-Hadamard basis coefficients.
        device: Device on which parameters and buffers are allocated
            (e.g. ``"cpu"`` or ``"cuda"``).
        grad_factor: Scaling factor applied to the gradient of the input using
            ``GradFactor``. A value of 1.0 leaves gradients unchanged.
        connections_method: Strategy for wiring input features to each neuron.
            Supported values are:
            - ``"random"``: Randomly sampled connections per neuron.
            - ``"random-unique"``: Random, non-overlapping connections
                (currently only for ``lut_rank == 2``).
        weight_init: Initialization scheme for LUT weights. Supported values:
            - ``"residual"``: Residual-style initialization around a default LUT.
            - ``"random"``: Fully random logits for each possible LUT.
        residual_probability: Scalar parameter controlling the strength of the
            residual initialization when ``weight_init == "residual"``.
        temperature: Temperature parameter used for (Gumbel-)Softmax/Sigmoid
            sampling of LUT weights during training.
        forward_sampling: Strategy for sampling LUT weights in the forward pass.
            Supported values:
            - ``"soft"``: Softmax/Sigmoid over weights (continuous relaxation).
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax/Sigmoid (continuous).
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax/Sigmoid.
        lut_rank: Number of inputs per logic gate (arity of the LUT). The
            number of possible entries in the LUT is ``2 ** lut_rank``.
        arbitrary_basis: If True, allows a non-hardcoded basis for LUT
            parametrization (e.g., an arbitrary Walsh basis). If False,
            uses a fixed / hardcoded basis.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cuda",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,
    ):
        super().__init__(
            device=device,
            grad_factor=grad_factor,
            lut_rank=lut_rank,
            parametrization=parametrization,
            parametrization_kwargs=parametrization_kwargs,
            connections=connections,
            connections_kwargs=connections_kwargs,
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self._init_weights(out_dim=out_dim)
        self.connections = self._init_connections()
        # Legacy attributes for compatibility
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_weights(self, out_dim):
        # Initialize weights using parametrization
        weights = self.parametrization.init_weights(
            out_dim, self.device
        )
        return torch.nn.Parameter(weights)

    def forward(self, x):
        """Applies the LogicDense transformation to the input.

        For each neuron, the layer:
        1. Selects ``lut_rank`` input features according to the connection
           pattern in ``self.indices``.
        2. Samples (or selects) LUT weights based on ``self.weight`` and
           the sampler strategy.
        3. Evaluates the resulting binary operation.

        Args:
            x: Input tensor of shape ``(..., in_dim)``. The last dimension must
                match ``self.in_dim``.

        Returns:
            A tensor of shape ``(..., out_dim)`` containing the neuron outputs.
        """
        assert x.ndim >= 2, x.ndim
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # Extract inputs according to connection pattern
        x = self.connections(x)  # Shape: (batch_size, lut_rank, out_dim)

        # Delegate to parametrization with einsum contraction
        # b=batch, n=neurons, k=num_basis/16
        return self.parametrization.forward(
            x, self.weight, self.training,
            contraction='bnk,nk->bn'
        )

    def extra_repr(self):
        """Returns a string representation for printing the module.

        Returns:
            A string summarizing the input dimension, output dimension, and
            whether the module is currently in training or evaluation mode.
        """
        return "{}, {}, {}".format(
            self.in_dim, self.out_dim, "train" if self.training else "eval"
        )

    def _init_connections(self):
        """Constructs input–neuron connection indices."""
        self.connections_kwargs["in_dim"] = self.in_dim
        self.connections_kwargs["out_dim"] = self.out_dim
        self.connections = setup_connections(
            structure="dense",
            connections=self.connections,
            lut_rank=self.lut_rank,
            device=self.device,
            **self.connections_kwargs
        )
        return self.connections
        
    def get_luts_and_ids(self, **kwargs):
        """Computes the most probable LUT and its ID for each neuron.

        Method is dependent on the chosen parametrization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each
                  entry is the integer ID of the corresponding LUT, obtained by
                  interpreting its truth table as a binary number (or None if
                  not applicable for high lut_rank).
        """
        return self.parametrization.get_luts_and_ids(self.weight, **kwargs)
    
    def get_luts(self, **kwargs):
        """Computes the most probable LUT for each neuron.

        Method is dependent on the chosen parametrization.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
        """
        return self.parametrization.get_luts(self.weight, **kwargs)

    def get_regularization_loss(self, regularizer: str):
        return get_regularization_loss(self.weight, regularizer)
    
    def rescale_weights(self, method):
        rescale_weights(self.weight, method)