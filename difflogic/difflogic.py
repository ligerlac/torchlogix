import warnings
import torch
import numpy as np
from .functional import bin_op_s, bin_op_cnn, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from rich import print
try:
    import difflogic_cuda
except ImportError:
    warnings.warn('failed to import difflogic_cuda. no cuda features will be available', ImportWarning)

########################################################################################################################

class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',

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
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
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
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
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
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)

        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)
        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
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
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., beta = 0., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        # print(f'x =\n{x}')
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau + self.beta

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################

class LogicCNNLayer(torch.nn.Module):
    def __init__(
            self,
            in_dim: _size_2_t,
            device: str = 'cuda',
            grad_factor: float = 1.,
            channels: int = 1,
            num_kernels: int = 16,
            tree_depth: int = None,
            receptive_field_size: int = None,
            implementation: str = None,
            connections: str = 'random',
            stride: int = None,
            padding: int = None
    ):
        super().__init__()
        # residual weights
        # self.tree_weights = []
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size ({receptive_field_size})."
        )

        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Clearer reverse iteration
            level_weights = torch.nn.ParameterList([
                torch.nn.Parameter(torch.randn(num_kernels, 16, device=device)) for _ in range(2 ** i)
            ])
            self.tree_weights.append(level_weights)
        self.in_dim = _pair(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        num_nodes = 2 ** self.tree_depth - 1
        self.channels = channels
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.padding = padding
        self.indices = [self.get_kernel_indices(self.num_kernels, receptive_field_size, padding, stride, device)]
        # Compute the remaining indices for the binary tree
        current_level_nodes = (self.tree_depth + 1)*2
        #assuming from the paper that it isn't randomly connected?
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=device)
            right_indices = torch.arange(1, size, 2, device=device)
            self.indices.append((left_indices, right_indices))


    def forward(self, x):
        current_level = x
        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_c = left_indices[..., 0], left_indices[..., 1], left_indices[..., 2]
        b_h, b_w, b_c = right_indices[..., 0], right_indices[..., 1], right_indices[..., 2]
        a = current_level[:, a_c, a_h, a_w]
        b = current_level[:, b_c, b_h, b_w]
        # Process first level
        level_weights = torch.stack(
            [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[0]], dim=0
        )  # Shape: [8, 16, 16]
        current_level = bin_op_cnn(a, b, level_weights)  # Shape: [100, 16, 576, 8]

        # Process remaining levels
        for level in range(1, self.tree_depth+1):
            left_indices, right_indices = self.indices[level]
            a = current_level[..., left_indices]
            b = current_level[..., right_indices]
            level_weights = torch.stack(
                [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[level]], dim=0
            #)  # Shape: [8, 16, 16]

            current_level = bin_op_cnn(a, b, level_weights)
        return current_level


    def get_kernel_indices(self, num_kernels, receptive_field_size, padding, stride, device='cuda'):
        sample_size = 2 ** self.tree_depth  # Number of random connections per kernel (binary tree depth)
        c, h, w = self.channels, self.in_dim[0], self.in_dim[1]  # Number of channels (C), and image dimensions (H, W)
        h_k, w_k = receptive_field_size, receptive_field_size  # Kernel height and width

        # Account for padding: increase the dimensions of the input image based on padding
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding


        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions ({h_padded}, {w_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to (with padding)
        h_starts = torch.arange(0, h_padded - h_k + 1, stride, device=device)  # Slide in height (stride=1)
        w_starts = torch.arange(0, w_padded - w_k + 1, stride, device=device)  # Slide in width (stride=1)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing='ij')

        # Lists to hold the final stacked results for all kernels
        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel
        for kernel_idx in range(num_kernels):
            # Randomly select `sample_size` positions within the receptive field for this kernel
            h_indices = torch.randint(0, h_k, (2*sample_size,), device=device)
            w_indices = torch.randint(0, w_k, (2*sample_size,), device=device)
            c_indices = torch.randint(0, c, (2*sample_size,), device=device)  # Random channel indices as well

            stacked_as = []
            stacked_bs = []

            # Now slide this kernel over the image (across all positions)
            for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
                # Get the receptive field indices
                h_grid_indices = h_indices + h_start  # Offsets for sliding the kernel
                w_grid_indices = w_indices + w_start  # Offsets for sliding the kernel
                c_grid_indices = c_indices  # No offset for channel, just use the random channel indices

                # Stack the indices for this position and this kernel
                indices = torch.stack([h_grid_indices, w_grid_indices, c_grid_indices], dim=-1)
                # Split the permuted indices for the binary tree (split the random connections)
                a, b = indices[:sample_size], indices[sample_size:]
                stacked_as.append(a)
                stacked_bs.append(b)

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)

class OrPoolingLayer(torch.nn.Module):
    # create layer that selects max in the kernel
    def __init__(self, kernel_size, stride, padding):
        super(OrPoolingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        num_kernels_each_direction = np.sqrt(x.shape[2])
        assert num_kernels_each_direction.is_integer(), num_kernels_each_direction
        x_reshaped = x.view(x.shape[0], x.shape[1], int(num_kernels_each_direction), int(num_kernels_each_direction))
        x = torch.nn.functional.max_pool2d(x_reshaped, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return x
