import numpy as np
import torch

from ..parametrization import RawLUTParametrization, WalshLUTParametrization
from ..sampling import SoftmaxSampler, SigmoidSampler
from ..functional import GradFactor, get_random_unique_connections
from ..packbitstensor import PackBitsTensor


class LogicDense(torch.nn.Module):
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
        residual_init_param: Scalar parameter controlling the strength of the
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
        residual_init_param: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.lut_rank = lut_rank

        # Create parametrization component
        if parametrization == "raw":
            self.parametrization = RawLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SoftmaxSampler(forward_sampling, temperature)
        elif parametrization == "walsh":
            self.parametrization = WalshLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SigmoidSampler(forward_sampling, temperature)
        else:
            raise ValueError(
                f"Unsupported parametrization: {parametrization}. "
                f"Choose 'raw' or 'walsh'."
            )

        # Initialize weights using parametrization
        weights = self.parametrization.init_weights(
            out_dim, weight_init, residual_init_param, device
        )
        self.weight = torch.nn.Parameter(weights)

        # Setup connections
        self.connections = connections
        assert connections in ["random", "random-unique"], (
            f"connections must be 'random' or 'random-unique', got {connections}"
        )
        self.indices = self._get_connections(connections)
        self.indices_T = self.indices.transpose(0, 1)

        # Legacy attributes for compatibility
        self.num_neurons = out_dim
        self.num_weights = out_dim

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
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # Delegate to parametrization
        return self.parametrization.forward_dense(
            x, self.indices, self.weight, self.sampler, self.training
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

    def _get_connections(self, connections):
        """Constructs input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features. This function returns a
        tensor encoding which input indices are connected to which neuron.

        Args:
            connections: Strategy for building connections. Supported values:
                - ``"random"``: Randomly sample ``lut_rank`` input indices for
                  each of the ``out_dim`` neurons.
                - ``"random-unique"``: Use a deterministic, non-overlapping pattern of
                  connections (currently only for ``lut_rank == 2``).

        Returns:
            A tensor of shape ``(lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.lut_rank, (
            f"Cannot have lut_rank > in_dim ({self.lut_rank} > {self.in_dim})"
        )

        if connections == "random":
            c = torch.randperm(self.lut_rank * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(self.lut_rank, self.out_dim)
            c = c.to(torch.int64).to(self.device)
            return c
        elif connections == "random-unique":
            return get_random_unique_connections(
                self.in_dim, self.out_dim, self.lut_rank, self.device
            )
        else:
            raise ValueError(connections)

    def get_lut_ids(self):
        """Computes the most probable LUT and its ID for each neuron.

        Returns the truth table by choosing the maximum weight over all
        possible Boolean functions (for raw parametrization) or by applying
        Walsh-Hadamard transform (for Walsh parametrization).

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
        return self.parametrization.get_lut_ids(self.weight)


##########################################################################
# CUDA Implementation (kept as subclass due to special CUDA operations)
##########################################################################


class LogicDenseCuda(LogicDense):
    """CUDA-optimized implementation of LogicDense.

    This class provides fast CUDA kernels for logic gate operations.
    It's only available when device='cuda'. For CPU, use the standard
    LogicDense implementation.

    The CUDA implementation is significantly faster (50-100x) but requires
    compiled CUDA extensions.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        connections: str = "random",
        weight_init: str = "residual",
        residual_init_param: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization=parametrization,
            device=device,
            grad_factor=grad_factor,
            connections=connections,
            weight_init=weight_init,
            residual_init_param=residual_init_param,
            temperature=temperature,
            forward_sampling=forward_sampling,
            lut_rank=lut_rank,
            arbitrary_basis=arbitrary_basis
        )

        # Additional indices for efficient CUDA backward pass
        given_x_indices_of_y = [[] for _ in range(in_dim)]
        indices_0_np = self.indices[0].cpu().numpy()
        indices_1_np = self.indices[1].cpu().numpy()
        for y in range(out_dim):
            given_x_indices_of_y[indices_0_np[y]].append(y)
            given_x_indices_of_y[indices_1_np[y]].append(y)

        self.given_x_indices_of_y_start = torch.tensor(
            np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(),
            device=device,
            dtype=torch.int64,
        )
        self.given_x_indices_of_y = torch.tensor(
            [item for sublist in given_x_indices_of_y for item in sublist],
            dtype=torch.int64,
            device=device,
        )

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, (
                "PackBitsTensor is not supported for the differentiable training mode."
            )
            assert self.device == "cuda", (
                f"PackBitsTensor is only supported for CUDA, not for {self.device}. "
                "If you want fast inference on CPU, please use CompiledDiffLogicModel."
            )
            return self.forward_cuda_eval(x)

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        return self.forward_cuda(x)

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == "cuda", x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weight, dim=-1).to(x.dtype)
            return LogicDenseCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weight.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicDenseCudaFunction.apply(
                    x, a, b, w,
                    self.given_x_indices_of_y_start,
                    self.given_x_indices_of_y,
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """Efficient batched evaluation using bit-packed tensors.

        WARNING: this is an in-place operation.

        Args:
            x: PackBitsTensor input.

        Returns:
            PackBitsTensor output.
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        import torchlogix_cuda

        a, b = self.indices
        w = self.weight.argmax(-1).to(torch.uint8)
        x.t = torchlogix_cuda.eval(x.t, a, b, w)

        return x


class LogicDenseCudaFunction(torch.autograd.Function):
    """Custom autograd function for CUDA-accelerated logic operations."""

    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        import torchlogix_cuda

        ctx.save_for_backward(
            x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y
        )
        return torchlogix_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        import torchlogix_cuda

        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torchlogix_cuda.backward_x(
                x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y
            )
        if ctx.needs_input_grad[3]:
            grad_w = torchlogix_cuda.backward_w(x, a, b, grad_y)

        return grad_x, None, None, grad_w, None, None, None
