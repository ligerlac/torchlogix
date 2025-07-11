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
        connections: str = "random",  # or 'unique'
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
        if connections == "random":
            self.indices = [
                self.get_kernel_indices(
                    self.num_kernels, receptive_field_size, padding, stride, device
                )
            ]
            # Compute the remaining indices for the binary tree
            # assuming from the paper that it isn't randomly connected?
            for level in range(self.tree_depth):
                size = 2 ** (self.tree_depth - level)
                left_indices = torch.arange(0, size, 2, device=device)
                right_indices = torch.arange(1, size, 2, device=device)
                self.indices.append((left_indices, right_indices))

        elif connections == "unique":
            # Create a full binary tree of all possible connections within the receptive field
            self.indices = [
                self.get_kernel_indices_unique(
                    num_kernels=1,
                    receptive_field_size=receptive_field_size,
                    padding=padding,
                    stride=stride,
                    device=device,
                )
            ]
            num_pairs = (receptive_field_size**2 * channels) * (receptive_field_size**2 * channels - 1) // 2
            assert num_pairs == self.indices[0][0].shape[-2], "Number of unique pairs does not match the expected number of pairs."
            next_power = 2 ** math.ceil(math.log2(num_pairs))
            print(f"Padding {num_pairs} pairs to {next_power} (next power of 2)")
        
            current_size = next_power
            
            for level in range(int(math.log2(next_power))):
                left_indices = torch.arange(0, current_size, 2, device=device)
                right_indices = torch.arange(1, current_size, 2, device=device)
                self.indices.append((left_indices, right_indices))
                current_size = current_size // 2


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

    def get_random_kernel_indices(
        self, num_kernels, receptive_field_size, padding, stride, device="cuda"
    ):
        """Get the indices of the kernels for the binary tree.
        this may contain self connections and duplicate connections.
        """
        # Number of random connections per kernel (binary tree depth)
        sample_size = 2**self.tree_depth
        # Number of channels (C), and image dimensions (H, W)
        c, h, w = self.channels, self.in_dim[0], self.in_dim[1]
        h_k, w_k = receptive_field_size, receptive_field_size  # Kernel height and width

        # Account for padding: increase the dimensions of the input image based on
        # padding
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding

        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to (with padding)
        h_starts = torch.arange(
            0, h_padded - h_k + 1, stride, device=device
        )  # Slide in height (stride=1)
        w_starts = torch.arange(
            0, w_padded - w_k + 1, stride, device=device
        )  # Slide in width (stride=1)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")

        # Lists to hold the final stacked results for all kernels
        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel
        for kernel_idx in range(num_kernels):
            # Randomly select `sample_size` positions within the receptive field for
            # this kernel
            h_indices = torch.randint(0, h_k, (2 * sample_size,), device=device)
            w_indices = torch.randint(0, w_k, (2 * sample_size,), device=device)
            # Random channel indices as well
            c_indices = torch.randint(0, c, (2 * sample_size,), device=device)

            stacked_as = []
            stacked_bs = []

            # Now slide this kernel over the image (across all positions)
            for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
                # Get the receptive field indices
                h_grid_indices = h_indices + h_start  # Offsets for sliding the kernel
                w_grid_indices = w_indices + w_start  # Offsets for sliding the kernel
                c_grid_indices = c_indices  # No offset for channel,
                # just use the random channel indices

                # Stack the indices for this position and this kernel
                indices = torch.stack(
                    [h_grid_indices, w_grid_indices, c_grid_indices], dim=-1
                )
                # Split the permuted indices for the binary tree (split the random
                # connections)
                a, b = indices[:sample_size], indices[sample_size:]
                stacked_as.append(a)
                stacked_bs.append(b)

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)


    def get_random_unique_kernel_indices(
            self, num_kernels, receptive_field_size, padding, stride, device="cuda"
        ):
        """Random connections within the receptive field, but unique pairs and no self-connections."""
        sample_size = 2**self.tree_depth
        c, h, w = self.channels, self.in_dim[0], self.in_dim[1]
        h_k, w_k = receptive_field_size, receptive_field_size
        
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding
        assert h_k <= h_padded and w_k <= w_padded
        
        num_positions = h_k * w_k * c
        max_unique_pairs = num_positions * (num_positions - 1) // 2
        
        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")
        
        # Generate sliding positions
        h_starts = torch.arange(0, h_padded - h_k + 1, stride, device=device)
        w_starts = torch.arange(0, w_padded - w_k + 1, stride, device=device)
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")
        
        # Pre-compute all RF positions
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        c_rf = torch.arange(0, c, device=device)
        
        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)
        
        all_stacked_as = []
        all_stacked_bs = []
        
        for kernel_idx in range(num_kernels):
            # Use torch.randperm for efficient unique sampling
            # Create all possible pair indices
            triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=device)
            total_pairs = triu_indices.shape[1]
            
            # Randomly select sample_size pairs
            selected_pair_indices = torch.randperm(total_pairs, device=device)[:sample_size]
            selected_i = triu_indices[0, selected_pair_indices]
            selected_j = triu_indices[1, selected_pair_indices]
            
            pairs_a = all_positions[selected_i]
            pairs_b = all_positions[selected_j]
            
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
            
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))
        
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)


    def get_kernel_indices_unique(
        self, num_kernels, receptive_field_size, padding, stride, device="cuda"
    ):
        """Get the indices of the kernels using all possible unique connections (pairs)."""
        # Number of channels (C), and image dimensions (H, W)
        c, h, w = self.channels, self.in_dim[0], self.in_dim[1]
        h_k, w_k = receptive_field_size, receptive_field_size  # Kernel height and width
        
        # Account for padding: increase the dimensions of the input image based on padding
        h_padded = h + 2 * padding
        w_padded = w + 2 * padding
        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )
        
        # Generate all possible positions the kernel can slide to (with padding)
        h_starts = torch.arange(0, h_padded - h_k + 1, stride, device=device)
        w_starts = torch.arange(0, w_padded - w_k + 1, stride, device=device)
        
        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")
        
        # Generate all possible positions within the receptive field
        h_rf_positions = torch.arange(0, h_k, device=device)
        w_rf_positions = torch.arange(0, w_k, device=device)
        c_positions = torch.arange(0, c, device=device)
        
        # Create all combinations of (height, width, channel) within receptive field
        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(
            h_rf_positions, w_rf_positions, c_positions, indexing="ij"
        )
        
        # Flatten to get all unique positions
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(), 
            c_rf_grid.flatten()
        ], dim=1)  # Shape: (num_positions, 3)
        
        num_positions = all_positions.shape[0]
        
        # Generate all unique pairs of positions (combinations, not permutations)
        # This creates C(num_positions, 2) pairs
        pairs_a = []
        pairs_b = []
        
        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                pairs_a.append(all_positions[i])
                pairs_b.append(all_positions[j])
        
        # Convert to tensors
        pairs_a = torch.stack(pairs_a, dim=0)  # Shape: (num_pairs, 3)
        pairs_b = torch.stack(pairs_b, dim=0)  # Shape: (num_pairs, 3)
        
        sample_size = len(pairs_a)  # Number of unique pairs
        
        print(f"Receptive field: {h_k}x{w_k}x{c}")
        print(f"Number of positions: {num_positions}")
        print(f"Number of unique pairs: {sample_size}")
        
        # Lists to hold the final stacked results for all kernels
        all_stacked_as = []
        all_stacked_bs = []
        
        # Process for each kernel (each kernel uses the same connection pattern)
        for kernel_idx in range(num_kernels):
            stacked_as = []
            stacked_bs = []
            
            # Now slide this kernel over the image (across all positions)
            for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
                # Get the receptive field indices for pairs_a
                h_grid_indices_a = pairs_a[:, 0] + h_start
                w_grid_indices_a = pairs_a[:, 1] + w_start
                c_grid_indices_a = pairs_a[:, 2]  # No offset for channel
                
                # Get the receptive field indices for pairs_b
                h_grid_indices_b = pairs_b[:, 0] + h_start
                w_grid_indices_b = pairs_b[:, 1] + w_start
                c_grid_indices_b = pairs_b[:, 2]  # No offset for channel
                
                # Stack the indices for this position and this kernel
                indices_a = torch.stack(
                    [h_grid_indices_a, w_grid_indices_a, c_grid_indices_a], dim=-1
                )
                indices_b = torch.stack(
                    [h_grid_indices_b, w_grid_indices_b, c_grid_indices_b], dim=-1
                )
                
                stacked_as.append(indices_a)
                stacked_bs.append(indices_b)
            
            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))
        
        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)


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
