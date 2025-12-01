import torch

from ..parametrization import RawLUTParametrization, WalshLUTParametrization
from ..functional import GradFactor, get_random_unique_connections
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
        connections: Strategy for wiring input features to each neuron.
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
        parametrization: str = "raw",
        device: str = "cpu",
        grad_factor: float = 1.0,
        connections: str = "random",
        weight_init: str = "residual",
        residual_probability: float = 0.9,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__(
            parametrization=parametrization, 
            device=device, 
            grad_factor=grad_factor, 
            temperature=temperature,
            forward_sampling=forward_sampling, 
            lut_rank=lut_rank, 
            arbitrary_basis=arbitrary_basis,
            connections=connections,
            weight_init=weight_init,
            residual_probability=residual_probability,
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self._init_weights(out_dim=out_dim)
        self.indices = self._init_connections()
        # Legacy attributes for compatibility
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_weights(self, out_dim):
        # Initialize weights using parametrization
        weights = self.parametrization.init_weights(
            out_dim, self.weight_init, self.residual_probability, self.device
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
        indices = self.indices.long()
        x = x[:, indices]  # Shape: (batch_size, lut_rank, out_dim)

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
        """Constructs input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features. This function returns a
        tensor encoding which input indices are connected to which neuron.

        Returns:
            A tensor of shape ``(lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.lut_rank, (
            f"Cannot have lut_rank > in_dim ({self.lut_rank} > {self.in_dim})"
        )

        if self.connections == "random":
            c = torch.randperm(self.lut_rank * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(self.lut_rank, self.out_dim)
            c = c.to(torch.int64).to(self.device)
            return c
        elif self.connections == "random-unique":
            return get_random_unique_connections(
                self.in_dim, self.out_dim, self.lut_rank, self.device
            )
        else:
            raise ValueError(self.connections)
        
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
