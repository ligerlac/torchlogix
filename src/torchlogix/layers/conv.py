import math
import random
from typing import Union

import torch
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from ..connections import setup_connections
from ..functional import (
    get_regularization_loss, rescale_weights
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

        Works with torch.Tensor (train/eval) and numpy (eval only).

        The forward pass proceeds as follows:

        1. Optionally pad the input spatially.
        2. Select all receptive-field positions for the first tree level using
           precomputed index tensors.
        3. For each tree level:
            a. In training: Sample LUT weights and apply with parametrization.
            b. In eval: Apply discrete LUTs using boolean operations.
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
        # Apply padding
        if self.padding > 0:
            if self.training:
                # Training: torch-specific
                x = torch.nn.functional.pad(
                    x,
                    (self.padding, self.padding, self.padding, self.padding, 0, 0),
                    mode="constant",
                    value=0
                )
            else:
                # Eval: type-generic
                from torchlogix.utils import pad_generic
                x = pad_generic(x, self.padding)

        if self.training:
            # Training path - existing implementation
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
        else:
            # Eval path - discrete LUT application
            from torchlogix.utils import stack_luts, apply_luts_conv_level

            # Get discrete LUT IDs for entire tree (configuration data)
            _, tree_ids = self.get_luts_and_ids()
            # Convert to numpy for indexing
            tree_ids = [[tid.cpu().detach().numpy() if hasattr(tid, 'cpu') else tid
                         for tid in level] for level in tree_ids]

            # Level 0: Select from receptive fields (type-agnostic!)
            selected = self.connections(x, tree_level=0)  # (batch, lut_rank, K, P, L)

            a = selected[:, 0]  # (batch, K, P, L)
            b = selected[:, 1]

            # Apply discrete LUTs at level 0
            lut_0 = stack_luts(tree_ids[0])
            batch, K, P, L = a.shape
            result = apply_luts_conv_level(a, b, lut_0, batch, K, P, L)

            # Process remaining tree levels
            for level in range(1, self.tree_depth):
                # Select using connections (type-agnostic!)
                selected = self.connections(result, tree_level=level)

                a = selected[..., 0, :]
                b = selected[..., 1, :]

                # Apply discrete LUTs for this level
                lut_level = stack_luts(tree_ids[level])
                batch, K, P, N = a.shape
                result = apply_luts_conv_level(a, b, lut_level, batch, K, P, N)

            # Calculate output dimensions and reshape
            out_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size[0]) // self.stride + 1
            out_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size[1]) // self.stride + 1

            result = result[..., 0]  # Take final tree output
            result = result.reshape(batch, self.num_kernels, out_h, out_w)

            return result

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
