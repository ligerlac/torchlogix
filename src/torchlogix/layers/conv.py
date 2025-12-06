import math
import random
from typing import Union

import torch
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from torchlogix.functional import (
    get_regularization_loss, rescale_weights, get_combination_indices
    )

from .base import LogicBase


class _LogicConvNd(LogicBase):
    """Abstract baseclass for convolutional logic layers."""

    def __init__(
        self,
        in_dim: Union[_size_2_t, _size_3_t, int],
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_2_t, _size_3_t, int] = 2,
        connections: str = "random",
        weight_init: str = "residual",
        stride: int = 1,
        padding: int = 0,
        conv_dimension: int = 2,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        residual_probability: float = 0.9,
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__(
            parametrization=parametrization, 
            device=device, 
            grad_factor=grad_factor, 
            temperature=temperature,
            forward_sampling=forward_sampling, 
            lut_rank=lut_rank, 
            arbitrary_basis=arbitrary_basis,
            connections=connections,
            weight_init=weight_init,
            residual_probability=residual_probability,
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
        self.indices = self._init_connections()
        
    def _init_weights(self):
        # Initialize tree weights using parametrization
        tree_weights = torch.nn.ParameterList()
        for i in reversed(range(self.tree_depth + 1)):
            # each tree level has lut_rank**i nodes per kernel
            level_weights = torch.nn.Parameter(torch.stack(
                [
                    self.parametrization.init_weights(
                        self.num_kernels, 
                        self.weight_init, 
                        self.residual_probability, 
                        self.device
                    ) for _ in range(self.lut_rank**i)
                ]
            ))
            tree_weights.append(level_weights)
        return tree_weights

    def _init_connections(self):
         # Setup connections
        if self.connections == "random":
            kernels = self._get_random_receptive_field_tensor()
        elif self.connections == "random-unique":
            kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {self.connections}")
        # Build tree indices
        return self._get_indices_from_kernel_tensor(kernels)

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
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding, 0, 0),
                mode="constant",
                value=0
            )
        # First level tree indices
        x = x[(slice(None), self.indices[0][..., -1], 
              *self.indices[0][..., :-1].moveaxis(-1, 0))]
        # Process first level with einsum contraction
        # b=batch, c=channels, s=spatial, f=features, k=num_basis/16
        x = self.parametrization.forward(
            x, self.tree_weights[0], self.training,
            contraction='bcsfk,fck->bcsf'
        )
        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            x = x.movedim(-2, 1)
            x = self.parametrization.forward(
                x, self.tree_weights[level], self.training,
                contraction='bcsfk,fck->bcsf'
            )
        # Reshape flattened output
        reshape = [(in_dim + 2*self.padding - rfs) // self.stride + 1 
                   for in_dim, rfs in zip(self.in_dim, self.receptive_field_size)]
        x = x.view(x.shape[0], x.shape[1], *reshape)

        return x

    def _get_random_receptive_field_tensor(self):
        """Generate random index tensor within the receptive field for each kernel.
        May contain self connections and duplicate connections.

        Returns:
            indices: (lut_rank, num_kernels, sample_size, 3)
                    where the last dim is (h, w, c)
        """
        c = self.channels
        sample_size = self.lut_rank ** self.tree_depth

        size = (self.num_kernels, self.lut_rank, sample_size)
        dim_indices = [torch.randint(0, dim, size, device=self.device) 
                       for dim in self.receptive_field_size]
        c_indices = torch.randint(0, c, size, device=self.device)

        # shape: (num_kernels, lut_rank, sample_size, dim)
        indices = torch.stack((*dim_indices, c_indices), dim=-1)

        return indices.transpose(0, 1)

    def _get_random_unique_receptive_field_tensor(self):
        """Generate random unique index tensor within the receptive field for each kernel.
        - No self-connections inside a tuple (all positions distinct).
        - No duplicate tuples within each kernel (unordered combinations).

        Returns:
            coords: tensor of shape (lut_rank, num_kernels, sample_size, 3)
                    coords[t, k, s] is the (h, w, c) of the t-th element of the s-th tuple
                    for kernel k.
        """
        c = self.channels
        sample_size = self.lut_rank ** self.tree_depth
        device = self.device

        # All RF positions as (h, w, c)
        rf = [torch.arange(0, dim, device=device) for dim in self.receptive_field_size]
        c_rf = torch.arange(0, c, device=device)

        grid = torch.meshgrid(*rf, c_rf, indexing="ij")
        all_positions = torch.stack([g.flatten() for g in grid], dim=1)

        num_positions = torch.prod(torch.tensor(self.receptive_field_size)) * c

        # All unique unordered index tuples of size `lut_rank`
        positions_1d = torch.arange(num_positions, device=device)
        comb_indices = get_combination_indices(
            n=num_positions,
            k=self.lut_rank,
            sample_size=sample_size,
            num_sets=self.num_kernels,
            device=self.device
        )
        tuple_indices = positions_1d[comb_indices] 
        # Map to (h, w, c) coordinates
        coords = all_positions[tuple_indices]  # (K, sample_size, lut_rank, 3)

        # Put tuple axis first
        coords = coords.permute(2, 0, 1, 3)  # (lut_rank, K, sample_size, 3)

        return coords

    def _apply_sliding_window_tensor(self, tensor):
        """Apply sliding window offsets to receptive field tensor.

        Args:
            tensor: torch.Tensor of shape (lut_rank, num_kernels, sample_size, 3)
                where last dim is (h, w, c).

        Returns:
            out: torch.Tensor of shape (lut_rank, num_kernels, num_positions, sample_size, 3),
                with the sliding-window offsets applied.
        """
        #h, w = self.in_dim
        #h_k, w_k = self.receptive_field_size

        # Account for padding
        padded = [in_dim + 2 * self.padding for in_dim in self.in_dim]
        #h_padded = h + 2 * self.padding
        #w_padded = w + 2 * self.padding

        assert all(rfs <= p for rfs, p in zip(self.receptive_field_size, padded)), (
            f"Receptive field size {self.receptive_field_size} must fit within input "
            f"dimensions {padded} after padding."
        )

        # Sliding positions
        starts = [torch.arange(0, p - rcf + 1, self.stride, device=self.device) 
                  for p, rcf in zip(padded, self.receptive_field_size)]
        #h_starts = torch.arange(0, padded[0] - self.receptive_field_size[0] + 1, self.stride, device=self.device)
        #w_starts = torch.arange(0, padded[1] - self.receptive_field_size[1] + 1, self.stride, device=self.device)

        # Meshgrid for all receptive-field start positions
        grid = torch.meshgrid(*starts, indexing="ij")
        offsets = [g.flatten() for g in grid]
        num_positions = [o.numel() for o in offsets]

        # tensor: (L, K, S, 3) → (K, L, S, 3)
        pairs_all = tensor.permute(1, 0, 2, 3)
        # K, L, S, _ = pairs_all.shape

        # Split h, w, c coordinates: (K, L, S)
        base = [pairs_all[..., i] for i in range(len(offsets))]
        #h_base = pairs_all[..., 0]
        #w_base = pairs_all[..., 1]
        c_base = pairs_all[..., -1]

        # Add sliding-window offsets (broadcasted) → (K, P, L, S)
        idx = [b.unsqueeze(1) + o.view(1, num_positions[0], 1, 1) 
               for b, o in zip(base, offsets)]
        c_idx = c_base.unsqueeze(1).expand(-1, num_positions[0], -1, -1)

        # Combine back into indices: (K, P, L, S, 3)
        all_indices = torch.stack([*idx, c_idx], dim=-1)

        # Reorder so first axis is L: (L, K, P, S, 3)
        out = all_indices.permute(2, 0, 1, 3, 4)

        return out

    def _get_indices_from_kernel_tensor(self, tensor):
        """Build index tensors for all tree levels."""
        indices = [
            self._apply_sliding_window_tensor(tensor)
        ]
        for level in range(self.tree_depth):
            size = self.lut_rank ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.lut_rank).transpose(0, 1)
            indices.append(base)
        return indices

    def get_luts_and_ids(self):
        """Computes the most probable LUT and its ID for each neuron.

        Returns:
            Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
                - ``tree_luts``: Nested list of Boolean tensors (truth tables)
                - ``tree_ids``: Nested list of integer tensors (LUT IDs)
        """
        tree_ids = []
        tree_luts = []
        for level in range(self.tree_depth + 1):
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
        for level in range(self.tree_depth + 1):
            level_luts = []
            for w in self.tree_weights[level]:
                luts = self.parametrization.get_luts(w)
                level_luts.append(luts)
            tree_luts.append(level_luts)
        return tree_luts
    
    def get_regularization_loss(self, regularizer: str):
        reg_loss = 0.0
        for w in self.tree_weights:
            reg_loss += get_regularization_loss(w, regularizer).sum(0)
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

    Args:
        in_dim: Input spatial dimensions ``(height, width)``.
        parametrization: LUT parametrization method. One of:
            - ``"raw"``: Direct truth table logits.
            - ``"walsh"``: Walsh-Hadamard basis coefficients.
            - ``"light"``: Light LGN parametrization.
        device: Device on which the module parameters and buffers are
            allocated (e.g. ``"cpu"`` or ``"cuda"``).
        grad_factor: Factor applied to the gradient of intermediate logic
            activations (e.g., via a custom autograd function) to control
            the strength of gradient flow.
        channels: Number of input channels.
        num_kernels: Number of output logic kernels (analogous to output
            channels in a standard convolution).
        tree_depth: Depth of the binary logic tree. A depth of ``d`` uses
            ``2**d`` leaves per receptive field.
        receptive_field_size: Spatial size (height and width) of the
            receptive field (assumed square).
        connections: Strategy for wiring the receptive field positions into
            the logic trees. Supported values:
            - ``"random"``: Randomly sampled receptive field positions for
                all leaves across kernels.
            - ``"random-unique"``: Randomly sampled non-repeating positions
                within each receptive field (unique connections).
        weight_init: Weight initialization scheme for the LUT logits at
            each tree node. Supported values:
            - ``"residual"``: Residual-style init around a default LUT.
            - ``"random"``: Unstructured random logits.
        stride: Convolution stride in both spatial dimensions.
        padding: Zero-padding applied symmetrically to height and width
            before selecting receptive fields.
        temperature: Temperature used by (Gumbel-)Softmax/Sigmoid sampling
            of LUT weights at each tree node.
        forward_sampling: Sampling strategy for LUT weights. One of:
            - ``"soft"``: Softmax/Sigmoid with continuous relaxation.
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax/Sigmoid relaxation.
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax/Sigmoid.
        residual_probability: Scalar controlling the strength of the
            residual-style initialization when ``weight_init == "residual"``.
        lut_rank: Arity of each logic gate (number of Boolean inputs per
            node). Typically 2 for binary trees.
        arbitrary_basis: If ``True``, allows more general bases for the LUT
            parametrization (e.g., Walsh), rather than a fixed hard-coded
            basis.
    """
    def __init__(
        self,
        in_dim: _size_2_t,
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_2_t, int] = 2,
        connections: str = "random",
        weight_init: str = "residual",
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        residual_probability: float = 0.9,
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__(
            in_dim=in_dim,
            parametrization=parametrization,
            device=device,
            grad_factor=grad_factor,
            channels=channels,
            num_kernels=num_kernels,
            tree_depth=tree_depth,
            receptive_field_size=receptive_field_size,
            connections=connections,
            weight_init=weight_init,
            stride=stride,
            padding=padding,
            temperature=temperature,
            forward_sampling=forward_sampling,
            residual_probability=residual_probability,
            lut_rank=lut_rank,
            arbitrary_basis=arbitrary_basis,
            conv_dimension=2,
        )


class LogicConv3d(_LogicConvNd):
    """3D convolutional layer with differentiable logic operations.

    This layer implements a 3D convolution where each output location is
    computed by evaluating a learned logic tree over a receptive field.
    Instead of linear filters, it uses a binary tree of differentiable
    logic operations (LUTs) applied to selected positions in the receptive
    field, per kernel and per spatial location.

    Args:
        in_dim: Input spatial dimensions ``(height, width, depth)``.
        parametrization: LUT parametrization method. One of:
            - ``"raw"``: Direct truth table logits.
            - ``"walsh"``: Walsh-Hadamard basis coefficients.
            - ``"light"``: Light LGN parametrization.
        device: Device on which the module parameters and buffers are
            allocated (e.g. ``"cpu"`` or ``"cuda"``).
        grad_factor: Factor applied to the gradient of intermediate logic
            activations (e.g., via a custom autograd function) to control
            the strength of gradient flow.
        channels: Number of input channels.
        num_kernels: Number of output logic kernels (analogous to output
            channels in a standard convolution).
        tree_depth: Depth of the binary logic tree. A depth of ``d`` uses
            ``2**d`` leaves per receptive field.
        receptive_field_size: Spatial size (height and width) of the
            receptive field (assumed square).
        connections: Strategy for wiring the receptive field positions into
            the logic trees. Supported values:
            - ``"random"``: Randomly sampled receptive field positions for
                all leaves across kernels.
            - ``"random-unique"``: Randomly sampled non-repeating positions
                within each receptive field (unique connections).
        weight_init: Weight initialization scheme for the LUT logits at
            each tree node. Supported values:
            - ``"residual"``: Residual-style init around a default LUT.
            - ``"random"``: Unstructured random logits.
        stride: Convolution stride in both spatial dimensions.
        padding: Zero-padding applied symmetrically to height and width
            before selecting receptive fields.
        temperature: Temperature used by (Gumbel-)Softmax/Sigmoid sampling
            of LUT weights at each tree node.
        forward_sampling: Sampling strategy for LUT weights. One of:
            - ``"soft"``: Softmax/Sigmoid with continuous relaxation.
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax/Sigmoid relaxation.
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax/Sigmoid.
        residual_probability: Scalar controlling the strength of the
            residual-style initialization when ``weight_init == "residual"``.
        lut_rank: Arity of each logic gate (number of Boolean inputs per
            node). Typically 2 for binary trees.
        arbitrary_basis: If ``True``, allows more general bases for the LUT
            parametrization (e.g., Walsh), rather than a fixed hard-coded
            basis.
    """
    def __init__(
        self,
        in_dim: _size_2_t,
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: Union[_size_2_t, int] = 2,
        connections: str = "random",
        weight_init: str = "residual",
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        residual_probability: float = 0.9,
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__(
            in_dim=in_dim,
            parametrization=parametrization,
            device=device,
            grad_factor=grad_factor,
            channels=channels,
            num_kernels=num_kernels,
            tree_depth=tree_depth,
            receptive_field_size=receptive_field_size,
            connections=connections,
            weight_init=weight_init,
            stride=stride,
            padding=padding,
            temperature=temperature,
            forward_sampling=forward_sampling,
            residual_probability=residual_probability,
            lut_rank=lut_rank,
            arbitrary_basis=arbitrary_basis,
            conv_dimension=3,
        )
