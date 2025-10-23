import warnings

import numpy as np
import torch
from rich import print
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.functional import gumbel_softmax, softmax
from itertools import product

from ..functional import (
    GradFactor,
    bin_op_s,
    get_unique_connections,
    gumbel_sigmoid,
    soft_raw,
    soft_walsh,
    hard_raw,
    hard_walsh,
    WALSH_COEFFICIENTS,
)
from ..packbitstensor import PackBitsTensor

# try:
from torchlogix import torchlogix_cuda
# except ImportError:
#     warnings.warn(
#         "failed to import torchlogix_cuda. no cuda features will be available",
#         ImportWarning,
#     )

##########################################################################

class LogicDense(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        implementation: str = None,
        connections: str = "random",
        weight_init: str = "residual",  # "residual" or "random"
        parametrization: str = "raw",  # standard or walsh or anf
        temperature: float = 1.0,
        forward_sampling: str = "soft"  # "soft", "hard", "gumbel_soft", "gumbel_hard"
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()

        self.parametrization = parametrization
        self.temperature = temperature
        self.forward_sampling = forward_sampling
        self.weight_init = weight_init

        if self.parametrization == "raw":
            if weight_init == "residual":
                # all weights to 0 except for weight number 3, which is set to 5
                weights = torch.zeros((out_dim, 16), device=device)
                weights[:, 3] = 5.0
                self.weight = torch.nn.parameter.Parameter(weights)
            elif weight_init == "random":
                self.weight = torch.nn.parameter.Parameter(
                    torch.randn(out_dim, 16, device=device)
                )
            else:
                raise ValueError(weight_init)
        elif self.parametrization in ["walsh", "anf"]:
            if weight_init == "residual":
                # chose randomly from walsh_coefficients, but prefer id=10
                walsh_coefficients_tensor = torch.tensor(list(WALSH_COEFFICIENTS.values()), device=device)
                weights = walsh_coefficients_tensor[
                    torch.randint(0, 16, (out_dim,), device=device)
                ]
                n = out_dim // 2
                # set half of weights to id=10 (pick index randomly)
                indices = torch.randperm(out_dim, device=device)
                weights[indices[:n]] = walsh_coefficients_tensor[10]
                self.weight = torch.nn.parameter.Parameter(weights)
            elif weight_init == "random":
                self.weight = torch.nn.parameter.Parameter(
                    torch.randn(out_dim, 4, device=device) * 0.1
                )
            else:
                raise ValueError(weight_init)
        else:
            raise ValueError(self.parametrization)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks
        2. To provide a CPU implementation of differentiable logic gate networks
        """
        self.implementation = implementation
        if self.implementation is None and device == "cuda":
            self.implementation = "cuda"
        elif self.implementation is None and device == "cpu":
            self.implementation = "python"
        assert self.implementation in ["cuda", "python"], self.implementation

        self.connections = connections
        assert self.connections in ["random", "unique"], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == "cuda":
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
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

        self.num_neurons = out_dim
        self.num_weights = out_dim

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

        if self.implementation == "cuda":
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)
        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.parametrization == "raw":
            if self.training:
                if self.forward_sampling == "soft":
                    x = soft_raw(self.weight, tau=self.temperature)
                elif self.forward_sampling == "hard":
                    x = hard_raw(self.weight, tau=self.temperature)
                elif self.forward_sampling == "gumbel_soft":
                    x = gumbel_softmax(self.weight, tau=self.temperature, hard=False)
                elif self.forward_sampling == "gumbel_hard":
                    x = gumbel_softmax(self.weight, tau=self.temperature, hard=True)
                x = bin_op_s(a, b, x)
            else:
                weights = torch.nn.functional.one_hot(self.weight.argmax(-1), 16).to(
                    torch.float32
                )
                x = bin_op_s(a, b, weights)
        elif self.parametrization == "walsh":
            A = 2 * a -1
            B = 2 * b -1
            basis = torch.stack([
                torch.ones_like(A),
                A,
                B,
                A*B
            ], dim=-1)
            x = (self.weight * basis).sum(dim=-1)
            if self.training:
                if self.forward_sampling == "soft":
                    x = soft_walsh(x, tau=self.temperature)
                elif self.forward_sampling == "hard":
                    x = hard_walsh(x, tau=self.temperature)
                elif self.forward_sampling == "gumbel_soft":
                    x = gumbel_sigmoid(x, tau=self.temperature, hard=False)
                elif self.forward_sampling == "gumbel_hard":
                    x = gumbel_sigmoid(x, tau=self.temperature, hard=True)
            else:
                x = (x > 0).to(torch.float32)
        return x

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

    def extra_repr(self):
        return "{}, {}, {}".format(
            self.in_dim, self.out_dim, "train" if self.training else "eval"
        )

    def get_connections(self, connections, device="cuda"):
        # assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
        #                                        'number of inputs ({}) because otherwise not all inputs could be ' \
        #                                        'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == "random":
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == "unique":
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    def get_gate_ids(self):
        """Computes most-probable gate for each learned set of weights.
        Returns tensor of most-probable gate IDs."""

        assert self.parametrization in ["raw", "walsh"], \
            f"Cannot compute gate IDs for parameterization={self.parameterization}"

        if self.parametrization=="walsh":
            n_inputs = 2
            n_rows = 2**n_inputs

            # generate all 2^n input combinations
            binary_inputs = torch.tensor(
                list(product([-1, 1], repeat=n_inputs)),
                dtype=torch.float32, 
                device=self.device
            ) # shape: (4, 2)

            # generate truth tables
            truth_tables = (
                (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
                (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1), (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)
            )
            truth_tables = torch.tensor(truth_tables, dtype=torch.float32, device=self.device)


            num_gates, num_coeffs = self.weight.shape
            assert num_coeffs == 1 + n_inputs + (n_inputs*(n_inputs-1))//2, \
                f"Unexpected param shape {self.weight.shape} for n_inputs={n_inputs}"
            
            # bias term
            linear_preds = self.weight[:, 0].unsqueeze(1).expand(-1, n_rows)  # shape: (16, 4)

            # add linear terms
            for i in range(n_inputs):
                linear_preds = linear_preds + self.weight[:, i+1].unsqueeze(1) * binary_inputs[:, i].unsqueeze(0)

            # add pairwise product terms
            idx = n_inputs+1
            for i in range(n_inputs):
                for j in range(i+1, n_inputs):
                    linear_preds += (self.weight[:, idx].unsqueeze(1) * binary_inputs[:, i].unsqueeze(0) 
                                    * binary_inputs[:, j].unsqueeze(0))
                    idx +=1

            preds = (linear_preds > 0.0).float() # shape: (16, 4)
            dists = (preds.unsqueeze(1) != truth_tables.unsqueeze(0)).sum(dim=-1)
            ids = dists.argmin(axis=1) # index of closest truth table
        
        else: 
            ids = self.weight.argmax(axis=1)

        return ids



##########################################################################


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
