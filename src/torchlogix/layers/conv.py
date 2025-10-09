from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple
from torch.nn.functional import gumbel_softmax

from ..functional import bin_op_cnn, bin_op_cnn_walsh, gumbel_sigmoid, soft_raw, soft_walsh, hard_raw, hard_walsh, WALSH_COEFFICIENTS


class LogicConv2d(nn.Module):
    """2d convolutional layer with differentiable logic operations.

    This layer implements a 2d convolution with differentiable logic operations.
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
        weight_init: str = "residual",  # "residual" or "random"
        stride: int = 1,
        padding: int = 0,
        parametrization: str = "raw", # or 'walsh'
        temperature: float = 1.0,
        forward_sampling: str = "soft" # or "hard", "gumbel_soft", or "gumbel_hard"
    ):
        """Initialize the 2d logic convolutional layer.

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
            parametrization: Parametrization to use ("raw" or "walsh")
        """
        super().__init__()
        self.parametrization = parametrization
        self.forward_sampling = forward_sampling

        # self.tree_weights = []
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})."
        )
        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(2**i):  # Iterate over nodes at this level
                if self.parametrization == "raw":
                    if weight_init == "residual":
                        weights = torch.zeros(
                            num_kernels, 16, device=device
                        )  # Initialize with zeros
                        weights[:, 3] = 5  # Set the fourth element (index 3) to 5
                    elif weight_init == "random":
                        weights = torch.randn(num_kernels, 16, device=device)
                elif self.parametrization == "walsh":
                    if weight_init == "residual":
                        # chose randomly from walsh_coefficients, but prefer id=10
                        walsh_coefficients_tensor = torch.tensor(list(WALSH_COEFFICIENTS.values()), device=device)
                        weights = walsh_coefficients_tensor[
                            torch.randint(0, 16, (num_kernels,), device=device)
                        ].clone()  # .clone() for safety
                        n = num_kernels // 2
                        # set half of weights to id=10 (pick index randomly)
                        indices = torch.randperm(num_kernels, device=device)
                        weights[indices[:n]] = walsh_coefficients_tensor[10]
                    elif weight_init == "random":
                        weights = torch.randn(num_kernels, 4, device=device) * 0.1
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
        if connections == "random":
            self.kernel_pairs = self.get_random_receptive_field_pairs()
        elif connections == "random-unique":
            self.kernel_pairs = self.get_random_unique_receptive_field_pairs()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        self.indices = self.get_indices_from_kernel_pairs(self.kernel_pairs)
        self.temperature = temperature


    def forward(self, x):
        """Implement the binary tree using the pre-selected indices."""
        current_level = x
        if self.padding > 0:
            current_level = torch.nn.functional.pad(
                current_level,
                (self.padding, self.padding, self.padding, self.padding, 0, 0),
                mode="constant",
                value=0
            )

        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_c = left_indices[..., 0], left_indices[..., 1], left_indices[..., 2]
        b_h, b_w, b_c = right_indices[..., 0], right_indices[..., 1], right_indices[..., 2]
        a = current_level[:, a_c, a_h, a_w]
        b = current_level[:, b_c, b_h, b_w]

        if self.parametrization == "raw":
            weighting_func = {
                "soft": soft_raw,
                "hard": hard_raw,
                "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
                "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
            }[self.forward_sampling]

            level_weights = torch.stack(
                [weighting_func(w) for w in self.tree_weights[0]], dim=0
            )
            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                    torch.float32
                )

            current_level = bin_op_cnn(a, b, level_weights)

            # Process remaining levels
            for level in range(1, self.tree_depth + 1):
                left_indices, right_indices = self.indices[level]
                a = current_level[..., left_indices]
                b = current_level[..., right_indices]
                level_weights = torch.stack(
                    [weighting_func(w) for w in self.tree_weights[level]], dim=0
                )
                if not self.training:
                    level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                        torch.float32
                    )

                current_level = bin_op_cnn(a, b, level_weights)

        elif self.parametrization == "walsh":
            level_weights = torch.stack([w for w in self.tree_weights[0]], dim=0)
            current_level = bin_op_cnn_walsh(a, b, level_weights)
            if self.training:
                if self.forward_sampling == "soft":
                    current_level = soft_walsh(current_level, tau=self.temperature)
                elif self.forward_sampling == "hard":
                    current_level = hard_walsh(current_level, tau=self.temperature)
                elif self.forward_sampling == "gumbel_soft":
                    current_level = gumbel_sigmoid(current_level, tau=self.temperature, hard=False)
                elif self.forward_sampling == "gumbel_hard":
                    current_level = gumbel_sigmoid(current_level, tau=self.temperature, hard=True)
            else:
                current_level = (current_level > 0).to(torch.float32)

            # Process remaining levels
            for level in range(1, self.tree_depth + 1):
                left_indices, right_indices = self.indices[level]
                a = current_level[..., left_indices]
                b = current_level[..., right_indices]
                # level_weights = self.tree_weights[level]
                level_weights = torch.stack([w for w in self.tree_weights[level]], dim=0)
                current_level = bin_op_cnn_walsh(a, b, level_weights)
                if self.training:
                    current_level = torch.sigmoid(current_level / self.temperature)
                else:
                    current_level = (current_level > 0).to(torch.float32)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size) // self.stride + 1

        current_level = current_level.view(
            current_level.shape[0],
            current_level.shape[1],
            reshape_h,
            reshape_w,
        )

        return current_level


    def get_random_receptive_field_pairs(self):
        """Generate random index pairs within the receptive field.
        May contain self connections and duplicate connections.
        """
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = 2**self.tree_depth

        # Randomly select positions within the receptive field
        h_indices = torch.randint(0, h_k, (2 * sample_size,), device=self.device)
        w_indices = torch.randint(0, w_k, (2 * sample_size,), device=self.device)
        c_indices = torch.randint(0, c, (2 * sample_size,), device=self.device)

        # Stack the indices
        indices = torch.stack([h_indices, w_indices, c_indices], dim=-1)

        # Split for binary tree (split the random connections)
        pairs_a = indices[:sample_size]
        pairs_b = indices[sample_size:]

        return pairs_a, pairs_b


    def get_random_unique_receptive_field_pairs(self):
        """Generate random unique index pairs within the receptive field.
        No self-connections or duplicate pairs.
        """
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = 2**self.tree_depth

        # Pre-compute all RF positions
        h_rf = torch.arange(0, h_k, device=self.device)
        w_rf = torch.arange(0, w_k, device=self.device)
        c_rf = torch.arange(0, c, device=self.device)

        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)

        num_positions = h_k * w_k * c
        max_unique_pairs = num_positions * (num_positions - 1) // 2

        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")

        # Use torch.randperm for efficient unique sampling
        # Create all possible pair indices
        triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=self.device)
        total_pairs = triu_indices.shape[1]

        # Randomly select sample_size pairs
        selected_pair_indices = torch.randperm(total_pairs, device=self.device)[:sample_size]
        selected_i = triu_indices[0, selected_pair_indices]
        selected_j = triu_indices[1, selected_pair_indices]

        pairs_a = all_positions[selected_i]
        pairs_b = all_positions[selected_j]

        return pairs_a, pairs_b


    def apply_sliding_window(self, pairs_tuple):
        """Apply sliding window to the receptive field pairs across all kernel positions."""
        pairs_a, pairs_b = pairs_tuple
        h, w = self.in_dim[0], self.in_dim[1]
        h_k, w_k = self.receptive_field_size, self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding

        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")

        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel
        for kernel_idx in range(self.num_kernels):
            stacked_as = []
            stacked_bs = []

            # Slide the kernel over the image (across all positions)
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

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)

    
    def get_indices_from_kernel_pairs(self, pairs_tuple):
        indices = [
            self.apply_sliding_window(pairs_tuple)
        ]
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=self.device)
            right_indices = torch.arange(1, size, 2, device=self.device)
            indices.append((left_indices, right_indices))
        return indices


class LogicConv3d(nn.Module):
    """3d convolutional layer with differentiable logic operations.

    This layer implements a 3d convolution with differentiable logic operations.
    It uses a binary tree structure to combine input features using logical
    operations.
    """

    def __init__(
        self,
        in_dim: _size_3_t,
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: _size_3_t = None,
        implementation: str = None,
        connections: str = "random",  # or 'random-unique'
        stride: int = 1,
        padding: int = None,
    ):
        """Initialize the 3d logic convolutional layer.

        Args:
            in_dim: Input dimensions (height, width, depth)
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

        self.receptive_field_size = _triple(receptive_field_size)
        assert (
            (stride <= self.receptive_field_size[0]) and
            (stride <= self.receptive_field_size[1]) and
            (stride <= self.receptive_field_size[2])), (
                f"Stride ({stride}) cannot be larger than receptive field size "
                f"({receptive_field_size})"
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
        self.in_dim = _triple(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.connections = connections
        if connections == "random":
            self.kernel_pairs = self.get_random_receptive_field_pairs()
        elif connections == "random-unique":
            self.kernel_pairs = self.get_random_unique_receptive_field_pairs()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        self.indices = self.get_indices_from_kernel_pairs(self.kernel_pairs)


    def forward(self, x):
        """Implement the binary tree using the pre-selected indices."""
        current_level = x
        # apply zero padding
        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_d, a_c = (
            left_indices[..., 0],
            left_indices[..., 1],
            left_indices[..., 2],
            left_indices[..., 3]
        )
        b_h, b_w, b_d, b_c = (
            right_indices[..., 0],
            right_indices[..., 1],
            right_indices[..., 2],
            right_indices[..., 3]
        )
        a = current_level[:, a_c, a_h, a_w, a_d]
        b = current_level[:, b_c, b_h, b_w, b_d]

        # Process first level
        level_weights = torch.stack(
            [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[0]],
            dim=0,
        )
        if not self.training:
            level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                torch.float32
            )

        current_level = bin_op_cnn(a, b, level_weights)

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
            )

            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                    torch.float32
                )

            current_level = bin_op_cnn(a, b, level_weights)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size[0]) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size[1]) // self.stride + 1
        reshape_d = (self.in_dim[2] + 2*self.padding - self.receptive_field_size[2]) // self.stride + 1

        current_level = current_level.view(
            current_level.shape[0],
            current_level.shape[1],
            reshape_h,
            reshape_w,
            reshape_d
        )

        return current_level


    def get_random_receptive_field_pairs(self):
        """Generate random index pairs within the receptive field.
        May contain self connections and duplicate connections.
        """
        c, h_k, w_k, d_k = (
            self.channels,
            self.receptive_field_size[0],
            self.receptive_field_size[1],
            self.receptive_field_size[2]
        )
        sample_size = 2**self.tree_depth

        # Randomly select positions within the receptive field
        h_indices = torch.randint(0, h_k, (2 * sample_size,), device=self.device)
        w_indices = torch.randint(0, w_k, (2 * sample_size,), device=self.device)
        d_indices = torch.randint(0, d_k, (2 * sample_size,), device=self.device)
        c_indices = torch.randint(0, c, (2 * sample_size,), device=self.device)

        # Stack the indices
        indices = torch.stack([h_indices, w_indices, d_indices, c_indices], dim=-1)

        # Split for binary tree (split the random connections)
        pairs_a = indices[:sample_size]
        pairs_b = indices[sample_size:]

        return pairs_a, pairs_b


    def get_random_unique_receptive_field_pairs(self):
        """Generate random unique index pairs within the receptive field.
        No self-connections or duplicate pairs.
        """
        c, h_k, w_k, d_k = (
            self.channels,
            self.receptive_field_size[0],
            self.receptive_field_size[1],
            self.receptive_field_size[2]
        )
        sample_size = 2**self.tree_depth

        # Pre-compute all RF positions
        h_rf = torch.arange(0, h_k, device=self.device)
        w_rf = torch.arange(0, w_k, device=self.device)
        d_rf = torch.arange(0, d_k, device=self.device)
        c_rf = torch.arange(0, c, device=self.device)

        h_rf_grid, w_rf_grid, d_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, d_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            d_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)

        num_positions = h_k * w_k * d_k * c
        max_unique_pairs = num_positions * (num_positions - 1) // 2

        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")

        # Use torch.randperm for efficient unique sampling
        # Create all possible pair indices
        triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=self.device)
        total_pairs = triu_indices.shape[1]

        # Randomly select sample_size pairs
        selected_pair_indices = torch.randperm(total_pairs, device=self.device)[:sample_size]
        selected_i = triu_indices[0, selected_pair_indices]
        selected_j = triu_indices[1, selected_pair_indices]

        pairs_a = all_positions[selected_i]
        pairs_b = all_positions[selected_j]

        return pairs_a, pairs_b


    def apply_sliding_window(self, pairs_tuple):
        """Apply sliding window to the receptive field pairs across all kernel positions."""
        pairs_a, pairs_b = pairs_tuple
        h, w, d = self.in_dim[0], self.in_dim[1], self.in_dim[2]
        h_k, w_k, d_k = self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        d_padded = d + 2 * self.padding

        assert (h_k <= h_padded and w_k <= w_padded) and d_k <= d_padded, (
            f"Receptive field size ({h_k}, {w_k}, {d_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}, {d_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)
        d_starts = torch.arange(0, d_padded - d_k + 1, self.stride, device=self.device)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid, d_grid = torch.meshgrid(h_starts, w_starts, d_starts, indexing="ij")

        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel
        for kernel_idx in range(self.num_kernels):
            stacked_as = []
            stacked_bs = []

            # Slide the kernel over the image (across all positions)
            for h_start, w_start, d_start in zip(h_grid.flatten(), w_grid.flatten(), d_grid.flatten()):
                # Apply sliding window offset
                indices_a = torch.stack([
                    pairs_a[:, 0] + h_start,
                    pairs_a[:, 1] + w_start,
                    pairs_a[:, 2] + d_start,
                    pairs_a[:, 3]
                ], dim=-1)

                indices_b = torch.stack([
                    pairs_b[:, 0] + h_start,
                    pairs_b[:, 1] + w_start,
                    pairs_b[:, 2] + d_start,
                    pairs_b[:, 3]
                ], dim=-1)

                stacked_as.append(indices_a)
                stacked_bs.append(indices_b)

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)


    def get_indices_from_kernel_pairs(self, pairs_tuple):
        indices = [
            self.apply_sliding_window(pairs_tuple)
        ]
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=self.device)
            right_indices = torch.arange(1, size, 2, device=self.device)
            indices.append((left_indices, right_indices))
        return indices


class OrPooling(torch.nn.Module):
    """Logic gate based pooling layer."""

    # create layer that selects max in the kernel

    def __init__(self, kernel_size, stride, padding=0):
        super(OrPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool the max value in the kernel."""
        if (x.dim() == 4):
            x = torch.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        elif (x.dim() == 5):
            x = torch.nn.functional.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            raise NotImplementedError(
                "OrPooling only implemented for input tensor with rank 4 or 5"
            )
        return x
