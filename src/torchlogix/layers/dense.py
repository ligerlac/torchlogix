
import numpy as np
import torch
from torch.nn.functional import gumbel_softmax

from .initialization import initialize_weights_raw, initialize_weights_walsh
from ..functional import (
    GradFactor,
    bin_op_s,
    get_random_unique_connections,
    gumbel_sigmoid,
    softmax,
    sigmoid,
    walsh_basis_hard,
    walsh_hadamard_transform
    )
from ..packbitstensor import PackBitsTensor


##########################################################################


def setup_dense_cls(parametrization: str) -> torch.nn.Module:
    """Factory function to select the appropriate LogicDense subclass.

    Args:
        parametrization: String specifying the LUT parametrization method.
            Supported values are:
            - ``"raw"``: Raw LUT-space logits.
            - ``"walsh"``: Walsh–Hadamard basis coefficients.

    Returns:
        The corresponding LogicDense subclass.
    """
    if parametrization == "raw":
        return LogicDense
    elif parametrization == "walsh":
        return LogicDenseWalsh
    else:
        raise ValueError(f"Unsupported parametrization: {parametrization}")


class LogicDense(torch.nn.Module):
    """
    The core module for Weightless Neural Networks. This baseclass provides the
    implementation of Differentiable Deep Logic Gate Networks.

    Args:
        in_dim: Number of input features (last dimension of the input tensor).
        out_dim: Number of output neurons (logical units).
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
        temperature: Temperature parameter used for (Gumbel-)Softmax sampling
            of LUT weights during training.
        forward_sampling: Strategy for sampling LUT weights in the forward pass.
            Supported values:
            - ``"soft"``: Softmax over weights (continuous relaxation).
            - ``"hard"``: Straight-through hard selection via Softmax.
            - ``"gumbel_soft"``: Gumbel-Softmax (continuous).
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax (discrete).
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
        device: str = "cpu",
        grad_factor: float = 1.0,
        connections: str = "random-unique",
        weight_init: str = "residual",  # "residual" or "random"
        residual_init_param: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",  # "soft", "hard", "gumbel_soft", "gumbel_hard"
        lut_rank: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    ):
        super().__init__()
        self.temperature = temperature
        self.forward_sampling = forward_sampling
        self.weight_init = weight_init
        self.residual_init_param = residual_init_param
        self.lut_rank = lut_rank
        self.lut_entries = 1 << lut_rank
        self.arbitrary_basis = arbitrary_basis
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        idx = torch.arange(self.lut_entries, device=self.device)
        bits = ((idx.unsqueeze(1) >> torch.arange(self.lut_rank, device=self.device)) & 1)
        bits = bits.flip(1).view(1, 1, self.lut_entries, self.lut_rank)
        self.register_buffer("bits", bits.to(torch.float))
        weights, forward_sampling_func = self._init_weights()
        self.weight = torch.nn.Parameter(weights)
        self.forward_sampling_func = forward_sampling_func
        self.connections = connections
        assert self.connections in ["random", "random-unique"], self.connections
        self.indices = self._get_connections(self.connections)
        self.indices_T = self.indices.transpose(0, 1)
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_weights(self):
        assert self.lut_rank == 2, "Raw parametrization only supports 2 inputs."
        weights = initialize_weights_raw(self.weight_init, self.out_dim, self.lut_rank, self.residual_init_param, self.device)
        forward_sampling_func = {
            "soft": lambda w: softmax(w, tau=self.temperature, hard=False),
            "hard": lambda w: softmax(w, tau=self.temperature, hard=True),
            "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]
        return weights, forward_sampling_func

    def forward(self, x):
        """Applies the LogicDense transformation to the input.

        For each neuron, the layer:
        1. Selects ``lut_rank`` input features according to the connection
           pattern in ``self.indices``.
        2. Samples (or selects) LUT weights based on ``self.weight`` and
           ``self.forward_sampling_func``.
        3. Evaluates the resulting binary operation via ``bin_op_s``.

        Args:
            x: Input tensor of shape ``(..., in_dim)``. The last dimension must
                match ``self.in_dim``.

        Returns:
            A tensor of shape ``(..., out_dim)`` containing the neuron outputs.
        """
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)
        if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)
        self.indices = self.indices.long()
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            w = self.forward_sampling_func(self.weight)
            x = bin_op_s(a, b, w)
        else:
            w = torch.nn.functional.one_hot(self.weight.argmax(-1), 1 << self.lut_entries).to(
                torch.float32
            )
            x = bin_op_s(a, b, w)
        return x

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
        if connections == "random":
            c = torch.randperm(self.lut_rank * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(self.lut_rank, self.out_dim)
            c = c.to(torch.int64).to(self.device)
            return c
        elif connections == "random-unique":
            return get_random_unique_connections(self.in_dim, self.out_dim, self.lut_rank, self.device)
        else:
            raise ValueError(connections)

    def get_lut_ids(self):
        """Computes the most probable LUT and its ID for each neuron by choosing maximum weight over all
        possible Boolean functions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each
                  entry is the integer ID of the corresponding LUT, obtained by
                  interpreting its truth table as a binary number.
        """
        ids = self.weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(1 << self.lut_rank, device=ids.device)) & 1).flip(1)
        return luts, ids
    

##########################################################################


class LogicDenseWalsh(LogicDense):
    """Differentiable LUT network using a Walsh–Hadamard parametrization.

    This subclass of :class:`LogicDense` replaces the raw LUT-space
    parametrization with a Walsh–Hadamard basis. Instead of assigning 
    logits to the LUT truth table directly, the weights represent Walsh 
    coefficients, which are transformed back into a
    Boolean truth table via a Walsh–Hadamard transform.

    This representation encourages smoother optimization and exploits structure
    inherent to Boolean functions (e.g., parity, correlations).

    Args:
        in_dim: Number of input features.
        out_dim: Number of output logic neurons.
        device: Device on which parameters are allocated.
        grad_factor: Gradient scaling factor applied to the input.
        connections: Connection pattern for inputs to neurons. Must be either
            ``"random"`` or ``"unique"``.
        weight_init: Weight initialization strategy. One of:
            - ``"residual"``: initialize near an identity-like Walsh vector.
            - ``"random"``: random Walsh coefficients.
        residual_init_param: Scalar controlling residual initialization scale.
        temperature: Temperature used in the sampling function.
        forward_sampling: Sampling strategy used in the forward pass.
            One of {``"soft"``, ``"hard"``, ``"gumbel_soft"``, ``"gumbel_hard"``}.
        lut_rank: Arity of the Boolean function (size of LUT input).
        arbitrary_basis: If ``True``, a flexible user-defined basis is used
            instead of a hardcoded Walsh basis.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        connections: str = "random",
        weight_init: str = "residual",  # "residual" or "random"
        residual_init_param: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",  # "soft", "hard", "gumbel_soft", "gumbel_hard"
        lut_rank: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    ):
        super().__init__(in_dim=in_dim, 
                         out_dim=out_dim, 
                         device=device, 
                         grad_factor=grad_factor,
                         connections=connections, 
                         weight_init=weight_init, 
                         residual_init_param=residual_init_param,
                         temperature=temperature, 
                         forward_sampling=forward_sampling, 
                         lut_rank=lut_rank,
                         arbitrary_basis=arbitrary_basis)

    def _init_weights(self):
        assert self.arbitrary_basis or (self.lut_rank in [2, 4, 6]), "Hard basis only supports n=2, n=4 and n=6 for Walsh parametrization."
        weights = initialize_weights_walsh(self.weight_init, self.out_dim, self.lut_rank, self.residual_init_param, self.device)
        forward_sampling_func = {
            "soft": lambda w: sigmoid(w, tau=self.temperature, hard=False),
            "hard": lambda w: sigmoid(w, tau=self.temperature, hard=True),
            "gumbel_soft": lambda w: gumbel_sigmoid(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_sigmoid(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]
        return weights, forward_sampling_func
    
    def forward(self, x):
        """Applies the Walsh-based logic transformation.

        The procedure is:

        1. Extract input slices for each neuron according to ``self.indices``.
        2. Convert inputs to ``{-1, +1}`` values so that Walsh basis functions
           correspond to parity correlations.
        3. Compute basis activations either using:
            - a fast hardcoded Walsh basis (for n=2,4,6), or
            - a generic basis using the precomputed ``self.bits`` tensor.
        4. Combine basis activations with Walsh coefficients.
        5. Perform differentiable sampling via sigmoid / Gumbel-sigmoid.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.

        Returns:
            Tensor of shape ``(..., out_dim)`` containing the neuron outputs.
        """
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)
        self.indices = self.indices.long()
        x = 1 - 2 * x
        if not self.arbitrary_basis:
            basis = walsh_basis_hard(x, self.indices, self.lut_rank)
        else:
            x = x[..., self.indices_T]
            bits = self.bits
            basis = (1 - bits + bits * x.unsqueeze(-2)).prod(dim=-1)
        x = (self.weight * basis).sum(dim=-1)
        if self.training:
            x = self.forward_sampling_func(-x)
        else:
            x = (x < 0).to(dtype=torch.float32)
        return x
    
    def get_lut_ids(self):
        """Returns the most probable LUT truth tables and their integer IDs.

        The Walsh coefficients are transformed into Boolean truth tables by
        applying a Walsh–Hadamard transform and checking the sign:

        - Negative output → Boolean `1`
        - Non-negative output → Boolean `0`

        This yields the most likely LUT implemented by each neuron.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2**lut_rank)``.
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each ID
                  is the binary encoding of the corresponding truth table
                  (MSB first).
        """
        luts = walsh_hadamard_transform(self.weight, self.lut_rank)
        luts = luts < 0
        if self.lut_rank <= 4:
            ids = 2 ** torch.arange((1 << self.lut_rank) - 1, -1, -1, device=luts.device)
            ids = (luts * ids.unsqueeze(0)).sum(dim=1)
        else:
            ids = None
        return luts, ids


##########################################################################


class LogicDenseCuda(LogicDense):
    """
    The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only
    available for device='cuda'. The `python` implementation exists for 2 reasons:
    1. To provide an easy-to-understand implementation of differentiable logic gate networks
    2. To provide a CPU implementation of differentiable logic gate networks
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        connections: str = "random",
        weight_init: str = "residual",  # "residual" or "random"
        residual_init_param: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",  # "soft", "hard", "gumbel_soft", "gumbel_hard"
        lut_rank: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    ):
        super().__init__(in_dim=in_dim, 
                         out_dim=out_dim, 
                         device=device, 
                         grad_factor=grad_factor,
                         connections=connections, 
                         weight_init=weight_init, 
                         residual_init_param=residual_init_param,
                         temperature=temperature, 
                         forward_sampling=forward_sampling, 
                         lut_rank=lut_rank,
                         arbitrary_basis=arbitrary_basis)
        # Defining additional indices for improving the efficiency 
        # of the backward of the CUDA implementation.
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
            assert (
                not self.training
            ), "PackBitsTensor is not supported for the differentiable training mode."
            assert self.device == "cuda", (
                "PackBitsTensor is only supported for CUDA, not for {}. "
                "If you want fast inference on CPU, please use CompiledDiffLogicModel."
                "".format(self.device)
            )

        else:
            if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)
        if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
        return self.forward_cuda(x)
        
    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == "cuda", x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

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
                    x,
                    a,
                    b,
                    w,
                    self.given_x_indices_of_y_start,
                    self.given_x_indices_of_y,
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weight.argmax(-1).to(torch.uint8)
        x.t = torchlogix_cuda.eval(x.t, a, b, w)

        return x
    

class LogicDenseCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(
            x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y
        )
        return torchlogix_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
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