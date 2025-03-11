import warnings
import torch
import numpy as np
from difflogic.functional import bin_op_s, bin_op_cnn, get_unique_connections, GradFactor
from difflogic.packbitstensor import PackBitsTensor

from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from rich import print
try:
    import difflogic_cuda
except ImportError:
    warnings.warn('failed to import difflogic_cuda. no cuda features will be available', ImportWarning)

class LogicConv3d(torch.nn.Module):
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
            stride: int = 1,
            padding: int = None
    ):
        super().__init__()
        # residual weights

        # self.tree_weights = []
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size ({receptive_field_size})."
        )
        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(2 ** i):  # Iterate over nodes at this level
                weights = torch.zeros(num_kernels, 16, device=device)  # Initialize with zeros
                weights[:, 3] = 5  # Set the fourth element (index 3) to 5
                level_weights.append(torch.nn.Parameter(weights))  # Wrap as a trainable parameter
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
            )  # Shape: [8, 16, 16]

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
