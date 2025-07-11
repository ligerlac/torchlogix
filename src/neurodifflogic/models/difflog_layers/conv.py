from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from neurodifflogic.difflogic.functional import bin_op_cnn


class LogicConv3d(nn.Module):
    """3D convolutional layer with differentiable logic operations.

    This layer implements a 3D convolution with differentiable logic operations.
    It uses a binary tree structure to combine input features using logical
    operations.
    """

    def __init__(
        self,
        in_dim: _size_2_t,
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: int = None,
        implementation: str = None,
        connections: str = "random",  # or 'random-unique'
        stride: int = 1,
        padding: int = None,
    ):
        """Initialize the 3D logic convolutional layer.

        Args:
            in_dim: Input dimensions (height, width)
            device: Device to run the layer on
            grad_factor: Gradient factor for the logic operations
            channels: Number of input channels
            num_kernels: Number of output kernels
            tree_depth: Depth of the binary tree
            receptive_field_size: Size of the receptive field
            implementation: Implementation type ("python" or "cuda")
            connections: Connection type: "random" or "unique". The latter will overwrite
                the tree_depth parameter and use a full binary tree of all possible connections
                within the receptive field.
            stride: Stride of the convolution
            padding: Padding of the convolution
        """
        super().__init__()
        # residual weights

        # self.tree_weights = []
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})."
        )
        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(2**i):  # Iterate over nodes at this level
                weights = torch.zeros(
                    num_kernels, 16, device=device
                )  # Initialize with zeros
                weights[:, 3] = 5  # Set the fourth element (index 3) to 5
                # Wrap as a trainable parameter
                level_weights.append(torch.nn.Parameter(weights))
            self.tree_weights.append(level_weights)
        self.in_dim = _pair(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.padding = padding
        self.connections = connections
        indeces_generator = KernelIndicesGenerator(
            channels=self.channels,
            in_dim=self.in_dim,
            tree_depth=self.tree_depth,
        )
        self.indices = [
            indeces_generator.get_kernel_indices(
                num_kernels=self.num_kernels,
                receptive_field_size=self.receptive_field_size,
                padding=self.padding,
                stride=self.stride,
                pair_strategy=connections,  # Use the connections strategy
                device=self.device,
            )
        ]
        # if connections == "random":
        #     self.indices = [
        #         self.get_random_kernel_indices(
        #             self.num_kernels, receptive_field_size, padding, stride, device
        #         )
        #     ]
        # elif connections == "random-unique":
        #     self.indices = [
        #         self.get_random_unique_kernel_indices(
        #             self.num_kernels, receptive_field_size, padding, stride, device
        #         )
        #     ]
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=device)
            right_indices = torch.arange(1, size, 2, device=device)
            self.indices.append((left_indices, right_indices))


    def forward(self, x):
        """Implement the binary tree using the pre-selected indices."""
        current_level = x
        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_c = left_indices[..., 0], left_indices[..., 1], left_indices[..., 2]
        b_h, b_w, b_c = (
            right_indices[..., 0],
            right_indices[..., 1],
            right_indices[..., 2],
        )
        a = current_level[:, a_c, a_h, a_w]
        b = current_level[:, b_c, b_h, b_w]
        # Process first level
        level_weights = torch.stack(
            [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[0]],
            dim=0,
        )  # Shape: [8, 16, 16]
        current_level = bin_op_cnn(a, b, level_weights)  # Shape: [100, 16, 576, 8]

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            left_indices, right_indices = self.indices[level]
            a = current_level[..., left_indices]
            b = current_level[..., right_indices]
            level_weights = torch.stack(
                [
                    torch.nn.functional.softmax(w, dim=-1)
                    for w in self.tree_weights[level]
                ],
                dim=0,
            )  # Shape: [8, 16, 16]

            current_level = bin_op_cnn(a, b, level_weights)
        return current_level


class OrPoolingLayer(torch.nn.Module):
    """Logic gate based pooling layer."""

    # create layer that selects max in the kernel

    def __init__(self, kernel_size, stride, padding):
        super(OrPoolingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool the max value in the kernel."""
        num_kernels_each_direction = np.sqrt(x.shape[2])
        assert num_kernels_each_direction.is_integer(), num_kernels_each_direction
        x_reshaped = x.view(
            x.shape[0],
            x.shape[1],
            int(num_kernels_each_direction),
            int(num_kernels_each_direction),
        )
        x = torch.nn.functional.max_pool2d(
            x_reshaped,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return x


class KernelIndicesGenerator:
    def __init__(self, channels, in_dim, tree_depth):
        self.channels = channels
        self.in_dim = in_dim
        self.tree_depth = tree_depth
    
    def _setup_sliding_window(self, receptive_field_size, padding, stride, device):
        """Common setup for sliding window calculations."""
        c, h, w = self.channels, self.in_dim[0], self.in_dim[1]
        h_k, w_k = receptive_field_size, receptive_field_size
        
        # Account for padding
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding
        
        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )
        
        # Generate all possible positions the kernel can slide to
        h_starts = torch.arange(0, h_padded - h_k + 1, stride, device=device)
        w_starts = torch.arange(0, w_padded - w_k + 1, stride, device=device)
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")
        
        return h_k, w_k, h_grid, w_grid
    
    def _generate_receptive_field_positions(self, h_k, w_k, device):
        """Generate all possible positions within the receptive field."""
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        c_rf = torch.arange(0, self.channels, device=device)
        
        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)
        
        return all_positions
    
    def _apply_sliding_window(self, pairs_a, pairs_b, h_grid, w_grid):
        """Apply sliding window to pairs and return stacked results."""
        stacked_as = []
        stacked_bs = []
        
        for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
            # Apply sliding window offset
            indices_a = torch.stack([
                pairs_a[:, 0] + h_start,
                pairs_a[:, 1] + w_start,
                pairs_a[:, 2]
            ], dim=-1)
            
            indices_b = torch.stack([
                pairs_b[:, 0] + h_start,
                pairs_b[:, 1] + w_start,
                pairs_b[:, 2]
            ], dim=-1)
            
            stacked_as.append(indices_a)
            stacked_bs.append(indices_b)
        
        return torch.stack(stacked_as, dim=0), torch.stack(stacked_bs, dim=0)
    
    def _generate_random_pairs(self, all_positions, sample_size, device):
        """Generate random pairs (may include duplicates and self-connections)."""
        num_positions = all_positions.shape[0]
        
        # Generate random indices for pairs
        indices_flat = torch.randint(0, num_positions, (2 * sample_size,), device=device)
        
        # Split into two groups
        pairs_a = all_positions[indices_flat[:sample_size]]
        pairs_b = all_positions[indices_flat[sample_size:]]
        
        return pairs_a, pairs_b
    
    def _generate_unique_random_pairs(self, all_positions, sample_size, device):
        """Generate unique random pairs (no duplicates, no self-connections)."""
        num_positions = all_positions.shape[0]
        max_unique_pairs = num_positions * (num_positions - 1) // 2
        
        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")
        
        # Use torch.triu_indices for efficient unique pair generation
        triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=device)
        total_pairs = triu_indices.shape[1]
        
        # Randomly select sample_size pairs
        selected_pair_indices = torch.randperm(total_pairs, device=device)[:sample_size]
        selected_i = triu_indices[0, selected_pair_indices]
        selected_j = triu_indices[1, selected_pair_indices]
        
        pairs_a = all_positions[selected_i]
        pairs_b = all_positions[selected_j]
        
        return pairs_a, pairs_b
    
    def get_kernel_indices(
            self, num_kernels, receptive_field_size, padding, stride, pair_strategy="random", device="cuda"
        ):
        """
        Unified method that takes a strategy for pair generation.
        
        Args:
            pair_strategy: "random" or "random-unique"
        """
        sample_size = 2**self.tree_depth
        
        # Common setup
        h_k, w_k, h_grid, w_grid = self._setup_sliding_window(
            receptive_field_size, padding, stride, device
        )
        all_positions = self._generate_receptive_field_positions(h_k, w_k, device)
        
        # Select pair generation strategy
        if pair_strategy == "random":
            pair_generator = self._generate_random_pairs
        elif pair_strategy == "random-unique":
            pair_generator = self._generate_unique_random_pairs
            # Validate for unique pairs
            kernel_volume = receptive_field_size ** 2 * self.channels
            assert kernel_volume * (kernel_volume - 1) / 2 >= sample_size, \
                f"Kernel volume is too small for unique pairs with tree depth {self.tree_depth}."
        else:
            raise ValueError(f"Unknown pair strategy: {pair_strategy}")
        
        all_stacked_as = []
        all_stacked_bs = []
        
        for kernel_idx in range(num_kernels):
            # Generate pairs using selected strategy
            pairs_a, pairs_b = pair_generator(all_positions, sample_size, device)
            
            # Apply sliding window (common for all strategies)
            stacked_a, stacked_b = self._apply_sliding_window(pairs_a, pairs_b, h_grid, w_grid)
            
            all_stacked_as.append(stacked_a)
            all_stacked_bs.append(stacked_b)
        
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)
