import math
import numpy as np
import random
from typing import Union

import torch
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from ..connections import setup_connections
from ..functional import (
    get_regularization_loss, rescale_weights, apply_luts_vectorized_export_mode
    )
from .base import LogicBase


class _LogicConvNd(LogicBase):
    """Abstract baseclass for convolutional logic layers.
    This module provides common functionality for 2D and 3D logic convolutional
    layers with differentiable learning.
    
    Args:
        in_dim: Input spatial dimensions ``(depth, height, width)``.
        channels: Number of input channels.
        num_kernels: Number of output logic kernels (analogous to output channels)
        tree_depth: Depth of the binary logic tree. A depth of ``d`` uses
            ``2**d`` leaves per receptive field.
        receptive_field_size: Spatial size (depth, height and width) of the
            receptive field (assumed cubic).
        stride: Convolution stride in all spatial dimensions.
        padding: Zero-padding applied symmetrically to depth, height and width
            before selecting receptive fields.
        conv_dimension: Dimension of the convolution (2 or 3).
        device (str): Device to run the layer on ('cpu' or 'cuda').
        grad_factor (float): Gradient scaling factor.
        lut_rank (int): Rank of the LUTs used in the layer.
        parametrization (str): Type of parametrization to use ('raw', 'warp', 'light').
        parametrization_kwargs (dict): Additional keyword arguments for parametrization.
        connections (str): Type of connections to use ('fixed', 'learnable', etc.).
        connections_kwargs (dict): Additional keyword arguments for connections."""

    def __init__(
        self,
        in_dim: Union[_size_2_t, _size_3_t, int],
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_2_t, _size_3_t, int] = 2,
        stride: int = 1,
        padding: int = 0,
        conv_dimension: int = 2,
        device: str = "cpu",
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
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.conv_dimension = conv_dimension
        assert conv_dimension in [2, 3], "conv_dimension must be 2 or 3"
        if conv_dimension == 2:
            self.receptive_field_size = _pair(receptive_field_size)
            self.in_dim = _pair(in_dim)
        else:
            self.receptive_field_size = _triple(receptive_field_size)
            self.in_dim = _triple(in_dim)
        assert (
            all(stride <= dim for dim in self.receptive_field_size)
        ), (
            f"Stride ({stride}) cannot be larger than "
            f"receptive field size ({receptive_field_size})"
        )        
        self.stride = stride
        self.padding = padding
        self.tree_weights = self._init_weights()
        self.connections = self._init_connections()
        
    def _init_weights(self):
        # Initialize tree weights using parametrization
        tree_weights = torch.nn.ParameterList()
        for i in reversed(range(self.tree_depth)):
            # each tree level has lut_rank**i nodes per kernel
            level_weights = torch.nn.Parameter(torch.stack(
                [
                    self.parametrization.init_weights(
                        self.num_kernels, 
                        self.device
                    ) for _ in range(self.lut_rank**i)
                ]
            ))
            tree_weights.append(level_weights)
        return tree_weights

    def _init_connections(self):
         # Setup connections
        self.connections = setup_connections(
            structure="conv",
            connections=self.connections,
            lut_rank=self.lut_rank,
            device=self.device,
            in_dim=self.in_dim,
            channels=self.channels,
            num_kernels=self.num_kernels,
            tree_depth=self.tree_depth,
            receptive_field_size=self.receptive_field_size,
            conv_dimension=self.conv_dimension,
            stride=self.stride,
            padding=self.padding,
            **self.connections_kwargs
        )
        return self.connections

    def forward(self, x):
        """Applies the logic convolution to the input.

        The forward pass proceeds as follows:

        1. Optionally pad the input spatially.
        2. Select all receptive-field positions for the first tree level using
           precomputed index tensors.
        3. For each tree level:
            a. Sample or select LUT weights for all nodes at that level.
            b. Apply binary logic operations to the child activations,
               reducing them up the tree.
        4. Reshape the final per-kernel outputs into a 4D tensor of shape
           ``(batch_size, num_kernels, out_height, out_width)``.

        Args:
            x: Input tensor of shape ``(batch_size, channels, height, width)``.

        Returns:
            Tensor of shape ``(batch_size, num_kernels, out_height, out_width)``,
            where ``out_height`` and ``out_width`` are determined by the
            convolution parameters:

            * ``out_height = (in_height + 2 * padding - receptive_field_size) // stride + 1``
            * ``out_width  = (in_width  + 2 * padding - receptive_field_size) // stride + 1``.
        """
        if self.export_mode:
            return self._forward_export_mode(x)
        
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding, 0, 0),
                mode="constant",
                value=0
            )
        # First level tree indices
        x = self.connections(x, 0)
        # Process first level with einsum contraction
        # b=batch, c=channels, s=spatial, f=features, k=num_basis/16
        x = self.parametrization.forward(
            x, self.tree_weights[0], self.training,
            contraction='fc,bcsf->bcsf'
        )
        # Process remaining levels
        for level in range(1, self.tree_depth):
            x = self.connections(x, level)
            x = x.movedim(-2, 1)
            x = self.parametrization.forward(
                x, self.tree_weights[level], self.training,
                contraction='fc,bcsf->bcsf'
            )
        # Reshape flattened output
        reshape = [(in_dim + 2*self.padding - rfs) // self.stride + 1 
                   for in_dim, rfs in zip(self.in_dim, self.receptive_field_size)]
        x = x.view(x.shape[0], x.shape[1], *reshape)

        return x
    
    def _forward_export_mode(self, x):
        is_numpy = isinstance(x, np.ndarray)

        # Padding
        if self.padding > 0:
            if is_numpy:
                pad_width = [(0, 0), (0, 0)]
                pad_width.extend([(self.padding, self.padding)] * self.conv_dimension)
                x = np.pad(x, pad_width, mode='constant', constant_values=0)
            else:
                x = torch.nn.functional.pad(
                    x,
                    (self.padding, self.padding, self.padding, self.padding, 0, 0),
                    mode="constant",
                    value=0
                )

        # First level
        x = self.connections(x, 0)
        a, b = x[:, 0], x[:, 1]
        lut_ids = getattr(self, f'_export_lut_ids_L0')
        if is_numpy and not isinstance(lut_ids, np.ndarray):
            lut_ids = lut_ids.detach().cpu().numpy()
            lut_ids_bc = lut_ids.T[np.newaxis, :, np.newaxis, :]
        else:
            lut_ids_bc = lut_ids.T.unsqueeze(0).unsqueeze(-2)

        x = apply_luts_vectorized_export_mode(a, b, lut_ids_bc)

        # Remaining levels
        for level in range(1, self.tree_depth):
            x = self.connections(x, level)
            if is_numpy:
                x = np.moveaxis(x, -2, 1)
            else:
                x = x.movedim(-2, 1)
            a, b = x[:, 0], x[:, 1]
            lut_ids = getattr(self, f'_export_lut_ids_L{level}')
            n_trailing = a.ndim - 2
            if is_numpy and not isinstance(lut_ids, np.ndarray):
                lut_ids = lut_ids.detach().cpu().numpy()
                lut_ids_bc = lut_ids.T.reshape(
                    1, lut_ids.T.shape[0], *((1,) * (n_trailing - 1)), lut_ids.T.shape[1]
                )
            else:
                lut_ids_bc = lut_ids.T
                for _ in range(n_trailing - 1):
                    lut_ids_bc = lut_ids_bc.unsqueeze(-2)
                lut_ids_bc = lut_ids_bc.unsqueeze(0)
            x = apply_luts_vectorized_export_mode(a, b, lut_ids_bc)


        # Final reshape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        reshape = [
            (in_dim + 2 * self.padding - rfs) // self.stride + 1
            for in_dim, rfs in zip(self.in_dim, self.receptive_field_size)
        ]
        if is_numpy:
            x = x.reshape(x.shape[0], x.shape[1], *reshape)
        else:
            x = x.view(x.shape[0], x.shape[1], *reshape)

        return x

    def get_luts_and_ids(self):
        """Computes the most probable LUT and its ID for each neuron.

        Returns:
            Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
                - ``tree_luts``: Nested list of Boolean tensors (truth tables)
                - ``tree_ids``: Nested list of integer tensors (LUT IDs)
        """
        tree_ids = []
        tree_luts = []
        for level in range(self.tree_depth):
            level_ids = []
            level_luts = []
            for w in self.tree_weights[level]:
                luts, ids = self.parametrization.get_luts_and_ids(w)
                level_ids.append(ids)
                level_luts.append(luts)
            tree_ids.append(level_ids)
            tree_luts.append(level_luts)
        return tree_luts, tree_ids
    
    def get_luts(self):
        """Computes the most probable LUT for each neuron.

        Returns:
           List[List[torch.Tensor]]: Nested list of Boolean tensors (LUTs)
        """
        tree_luts = []
        for level in range(self.tree_depth):
            level_luts = []
            for w in self.tree_weights[level]:
                luts = self.parametrization.get_luts(w)
                level_luts.append(luts)
            tree_luts.append(level_luts)
        return tree_luts
    
    def get_regularization_loss(self, regularizer: str):
        reg_loss = 0.0
        for w in self.tree_weights:
            reg_loss += get_regularization_loss(w, regularizer)
        return reg_loss
    
    def rescale_weights(self, method):
        for w in self.tree_weights:
            rescale_weights(w, method)

    def set_export_mode(self, enabled: bool = True):
        self.eval()
        self.export_mode = enabled

        if enabled:
            _, tree_ids = self.get_luts_and_ids()

            # Stack each level (already same shape within level)
            for level_idx, level_ids in enumerate(tree_ids):
                # level_ids is list of tensors, all shape (num_kernels,)
                stacked = torch.stack(level_ids)  # Shape: (lut_rank**i, num_kernels)
                self.register_buffer(f'_export_lut_ids_L{level_idx}',
                                    stacked, persistent=True)
        else:
            # Clean up all buffers
            buffers_to_delete = [name for name in self._buffers.keys()
                                if name.startswith('_export_lut')]
            for name in buffers_to_delete:
                delattr(self, name)

    def _get_export_lut_ids(self, level):
        tree_ids = []
        for level_idx in range(self.tree_depth):
            # Just unstack (or iterate over dimension 0)
            level_tensor = getattr(self, f'_export_lut_ids_L{level_idx}')
            level_ids = [level_tensor[i] for i in range(level_tensor.shape[0])]
            tree_ids.append(level_ids)
        return tree_ids


class LogicConv2d(_LogicConvNd):
    """2D convolutional layer with differentiable logic operations.

    This layer implements a 2D convolution where each output location is
    computed by evaluating a learned logic tree over a receptive field.
    Instead of linear filters, it uses a binary tree of differentiable
    logic operations (LUTs) applied to selected positions in the receptive
    field, per kernel and per spatial location.
    """
    def __init__(
        self,
        in_dim: Union[_size_2_t, int],
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_2_t, int] = 2,
        stride: int = 1,
        padding: int = 0,
        device: str = "cpu",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,
    ):
        super().__init__(
            in_dim=in_dim,
            channels=channels,
            num_kernels=num_kernels,
            tree_depth=tree_depth,
            receptive_field_size=receptive_field_size,
            stride=stride,
            padding=padding,
            conv_dimension=2,
            device=device,
            grad_factor=grad_factor,
            lut_rank=lut_rank,
            parametrization=parametrization,
            parametrization_kwargs=parametrization_kwargs,
            connections=connections,
            connections_kwargs=connections_kwargs,
        )


class LogicConv3d(_LogicConvNd):
    """3D convolutional layer with differentiable logic operations.

    This layer implements a 3D convolution where each output location is
    computed by evaluating a learned logic tree over a receptive field.
    Instead of linear filters, it uses a binary tree of differentiable
    logic operations (LUTs) applied to selected positions in the receptive
    field, per kernel and per spatial location.
    """
    def __init__(
        self,
        in_dim: Union[_size_3_t, int],
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_3_t, int] = 2,
        stride: int = 1,
        padding: int = 0,
        device: str = "cpu",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,
    ):
        super().__init__(
            in_dim=in_dim,
            channels=channels,
            num_kernels=num_kernels,
            tree_depth=tree_depth,
            receptive_field_size=receptive_field_size,
            stride=stride,
            padding=padding,
            conv_dimension=3,
            device=device,
            grad_factor=grad_factor,
            lut_rank=lut_rank,
            parametrization=parametrization,
            parametrization_kwargs=parametrization_kwargs,
            connections=connections,
            connections_kwargs=connections_kwargs,
        )
