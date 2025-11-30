from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from ..parametrization import RawLUTParametrization, WalshLUTParametrization
from ..sampling import SoftmaxSampler, SigmoidSampler


class LogicConv2d(nn.Module):
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
        residual_init_param: Scalar controlling the strength of the
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
        receptive_field_size: int = None,
        connections: str = "random",
        weight_init: str = "residual",
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        residual_init_param: float = 1.0,
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__()
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})."
        )

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
        self.lut_rank = lut_rank

        # Create parametrization component
        if parametrization == "raw":
            self.parametrization = RawLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SoftmaxSampler(forward_sampling, temperature)
        elif parametrization == "walsh":
            self.parametrization = WalshLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SigmoidSampler(forward_sampling, temperature)
        else:
            raise ValueError(
                f"Unsupported parametrization: {parametrization}. "
                f"Choose 'raw' or 'walsh'."
            )

        # Setup connections
        if connections == "random":
            self.kernels = self._get_random_receptive_field_tensor()
        elif connections == "random-unique":
            self.kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {connections}")

        # Build tree indices
        self.indices = self._get_indices_from_kernel_tensor(self.kernels)

        # Initialize tree weights using parametrization
        self.tree_weights = torch.nn.ParameterList()
        for i in reversed(range(tree_depth + 1)):
            # each tree level has lut_rank**i nodes per kernel
            level_weights = torch.nn.Parameter(torch.stack(
                [
                    self.parametrization.init_weights(
                        num_kernels, weight_init, residual_init_param, device
                    ) for _ in range(lut_rank**i)
                ]
            ))
            self.tree_weights.append(level_weights)


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
        ind_h, ind_w, ind_c = self.indices[0][..., 0], self.indices[0][..., 1], self.indices[0][..., 2]
        x = x[:, ind_c, ind_h, ind_w]

        # Process first level
        x = self.parametrization.forward(
            x, self.tree_weights[0], self.sampler, self.training
        )

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            x = x.movedim(-2, 1)
            x = self.parametrization.forward(
                x, self.tree_weights[level], self.sampler, self.training
            )

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size) // self.stride + 1

        x = x.view(x.shape[0], x.shape[1], reshape_h, reshape_w)

        return x

    def _get_random_receptive_field_tensor(self):
        """Generate random index tensor within the receptive field for each kernel.
        May contain self connections and duplicate connections.

        Returns:
            indices: (lut_rank, num_kernels, sample_size, 3)
                    where the last dim is (h, w, c)
        """
        c = self.channels
        h_k = self.receptive_field_size
        w_k = self.receptive_field_size
        sample_size = self.lut_rank ** self.tree_depth

        size = (self.num_kernels, self.lut_rank, sample_size)

        h_indices = torch.randint(0, h_k, size, device=self.device)
        w_indices = torch.randint(0, w_k, size, device=self.device)
        c_indices = torch.randint(0, c, size, device=self.device)

        # shape: (num_kernels, lut_rank, sample_size, 3)
        indices = torch.stack((h_indices, w_indices, c_indices), dim=-1)

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
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = self.lut_rank ** self.tree_depth
        device = self.device

        # All RF positions as (h, w, c)
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        c_rf = torch.arange(0, c, device=device)

        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack(
            [h_rf_grid.flatten(), w_rf_grid.flatten(), c_rf_grid.flatten()],
            dim=1,
        )

        num_positions = h_k * w_k * c

        # All unique unordered index tuples of size `lut_rank`
        positions_1d = torch.arange(num_positions, device=device)
        comb = torch.combinations(positions_1d, r=self.lut_rank, with_replacement=False)
        total_tuples = comb.shape[0]

        if sample_size > total_tuples:
            raise ValueError(
                f"Not enough unique {self.lut_rank}-tuples: need {sample_size}, have {total_tuples}"
            )

        K = self.num_kernels

        # Sample `sample_size` tuples per kernel, without replacement
        selected_idx = torch.multinomial(
            torch.ones(K, total_tuples, device=device),
            sample_size,
            replacement=False
        )

        # Selected tuples of indices into all_positions: (K, sample_size, lut_rank)
        tuple_indices = comb[selected_idx]

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
        h, w = self.in_dim[0], self.in_dim[1]
        h_k, w_k = self.receptive_field_size, self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding

        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )

        # Sliding positions
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)

        # Meshgrid for all receptive-field start positions
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")
        h_offsets = h_grid.flatten()
        w_offsets = w_grid.flatten()
        num_positions = h_offsets.numel()

        # tensor: (L, K, S, 3) → (K, L, S, 3)
        pairs_all = tensor.permute(1, 0, 2, 3)
        K, L, S, _ = pairs_all.shape

        # Split h, w, c coordinates: (K, L, S)
        h_base = pairs_all[..., 0]
        w_base = pairs_all[..., 1]
        c_base = pairs_all[..., 2]

        # Add sliding-window offsets (broadcasted) → (K, P, L, S)
        h_idx = h_base.unsqueeze(1) + h_offsets.view(1, num_positions, 1, 1)
        w_idx = w_base.unsqueeze(1) + w_offsets.view(1, num_positions, 1, 1)
        c_idx = c_base.unsqueeze(1).expand(-1, num_positions, -1, -1)

        # Combine back into indices: (K, P, L, S, 3)
        all_indices = torch.stack([h_idx, w_idx, c_idx], dim=-1)

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

    def get_lut_ids(self):
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
                luts, ids = self.parametrization.get_lut_ids(w)
                level_ids.append(ids)
                level_luts.append(luts)
            tree_ids.append(level_ids)
            tree_luts.append(level_luts)
        return tree_luts, tree_ids


class LogicConv3d(nn.Module):
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
        receptive_field_size: Spatial size (height, width and depth) of the
            receptive field (assumed cubic).
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
        padding: Zero-padding applied symmetrically to height, width, and depth
            before selecting receptive fields.
        temperature: Temperature used by (Gumbel-)Softmax/Sigmoid sampling of LUT
            weights at each tree node.
        forward_sampling: Sampling strategy for LUT weights. One of:
            - ``"soft"``: Softmax/Sigmoid with continuous relaxation.
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax/Sigmoid relaxation.
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax/Sigmoid.
        residual_init_param: Scalar controlling the strength of the
            residual-style initialization when ``weight_init == "residual"``.
        lut_rank: Arity of each logic gate (number of Boolean inputs per
            node). Typically 2 for binary trees.
        arbitrary_basis: If ``True``, allows more general bases for the LUT
            parametrization (e.g., Walsh), rather than a fixed hard-coded
            basis.
    """

    def __init__(
        self,
        in_dim: _size_3_t,
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: _size_3_t = None,
        connections: str = "random",
        weight_init: str = "residual",
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        residual_init_param: float = 1.0,
        lut_rank: int = 2,
        arbitrary_basis: bool = False
    ):
        super().__init__()
        self.receptive_field_size = _triple(receptive_field_size)
        self.in_dim = _triple(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.connections = connections
        self.lut_rank = lut_rank

        assert (
            (stride <= self.receptive_field_size[0]) and
            (stride <= self.receptive_field_size[1]) and
            (stride <= self.receptive_field_size[2])
        ), (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})"
        )

        # Create parametrization component
        if parametrization == "raw":
            self.parametrization = RawLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SoftmaxSampler(forward_sampling, temperature)
        elif parametrization == "walsh":
            self.parametrization = WalshLUTParametrization(lut_rank, arbitrary_basis)
            self.sampler = SigmoidSampler(forward_sampling, temperature)
        else:
            raise ValueError(
                f"Unsupported parametrization: {parametrization}. "
                f"Choose 'raw' or 'walsh'."
            )

        # Setup connections
        if connections == "random":
            self.kernels = self._get_random_receptive_field_tensor()
        elif connections == "random-unique":
            self.kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {connections}")

        self.indices = self._get_indices_from_kernel_tensor(self.kernels)

        # Initialize tree weights using parametrization
        self.tree_weights = torch.nn.ParameterList()
        for i in reversed(range(tree_depth + 1)):
            # each tree level has lut_rank**i nodes per kernel
            level_weights = torch.nn.Parameter(torch.stack(
                [
                    self.parametrization.init_weights(
                        num_kernels, weight_init, residual_init_param, device
                    ) for _ in range(lut_rank**i)
                ]
            ))
            self.tree_weights.append(level_weights)            


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
        4. Reshape the final per-kernel outputs into a 5D tensor of shape
           ``(batch_size, num_kernels, out_height, out_width, out_depth)``.

        Args:
            x: Input tensor of shape ``(batch_size, channels, height, width, depth)``.

        Returns:
            Tensor of shape ``(batch_size, num_kernels, out_height, out_width, out_depth)``,
            where ``out_height``, ``out_width``, and ``out_depth`` are determined by the
            convolution parameters.
        """
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding),
                mode="constant",
                value=0
            )

        # First level
        ind_h, ind_w, ind_d, ind_c = (
            self.indices[0][..., 0], self.indices[0][..., 1],
            self.indices[0][..., 2], self.indices[0][..., 3]
        )
        x = x[:, ind_c, ind_h, ind_w, ind_d]

        # Process first level
        x = self.parametrization.forward(
            x, self.tree_weights[0], self.sampler, self.training
        )

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            x = x.movedim(-2, 1)
            x = self.parametrization.forward(
                x, self.tree_weights[level], self.sampler, self.training
            )

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size[0]) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size[1]) // self.stride + 1
        reshape_d = (self.in_dim[2] + 2*self.padding - self.receptive_field_size[2]) // self.stride + 1

        x = x.view(x.shape[0], x.shape[1], reshape_h, reshape_w, reshape_d)

        return x

    def _get_random_receptive_field_tensor(self):
        """Generate random index tensor within the receptive field for each kernel.
        May contain self connections and duplicate connections.

        Returns:
            indices: (lut_rank, num_kernels, sample_size, 4)
                    where the last dim is (h, w, d, c)
        """
        c = self.channels
        h_k, w_k, d_k = self.receptive_field_size
        sample_size = self.lut_rank ** self.tree_depth

        size = (self.num_kernels, self.lut_rank, sample_size)

        h_indices = torch.randint(0, h_k, size, device=self.device)
        w_indices = torch.randint(0, w_k, size, device=self.device)
        d_indices = torch.randint(0, d_k, size, device=self.device)
        c_indices = torch.randint(0, c, size, device=self.device)

        # shape: (num_kernels, lut_rank, sample_size, 4)
        indices = torch.stack((h_indices, w_indices, d_indices, c_indices), dim=-1)
        return indices.transpose(0, 1)

    def _get_random_unique_receptive_field_tensor(self):
        """Generate random unique index tensor within the receptive field for each kernel.
        - No self-connections inside a tuple (all positions distinct).
        - No duplicate tuples within each kernel (unordered combinations).

        Returns:
            coords: tensor of shape (lut_rank, num_kernels, sample_size, 4)
                    coords[t, k, s] is the (h, w, d, c) of the t-th element of the s-th tuple
                    for kernel k.
        """
        c = self.channels
        h_k, w_k, d_k = self.receptive_field_size
        sample_size = self.lut_rank ** self.tree_depth
        device = self.device

        # All RF positions as (h, w, d, c)
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        d_rf = torch.arange(0, d_k, device=device)
        c_rf = torch.arange(0, c, device=device)

        h_rf_grid, w_rf_grid, d_rf_grid, c_rf_grid = torch.meshgrid(
            h_rf, w_rf, d_rf, c_rf, indexing="ij"
        )
        all_positions = torch.stack(
            [h_rf_grid.flatten(), w_rf_grid.flatten(), d_rf_grid.flatten(), c_rf_grid.flatten()],
            dim=1,
        )

        num_positions = h_k * w_k * d_k * c

        # All unique unordered index tuples of size `lut_rank`
        positions_1d = torch.arange(num_positions, device=device)
        comb = torch.combinations(positions_1d, r=self.lut_rank, with_replacement=False)
        total_tuples = comb.shape[0]

        if sample_size > total_tuples:
            raise ValueError(
                f"Not enough unique {self.lut_rank}-tuples: need {sample_size}, have {total_tuples}"
            )

        K = self.num_kernels

        # Sample `sample_size` tuples per kernel, without replacement
        selected_idx = torch.multinomial(
            torch.ones(K, total_tuples, device=device),
            sample_size,
            replacement=False
        )

        # Selected tuples of indices into all_positions: (K, sample_size, lut_rank)
        tuple_indices = comb[selected_idx]

        # Map to (h, w, d, c) coordinates
        coords = all_positions[tuple_indices]  # (K, sample_size, lut_rank, 4)

        # Put tuple axis first
        coords = coords.permute(2, 0, 1, 3)  # (lut_rank, K, sample_size, 4)
        return coords

    def _apply_sliding_window_tensor(self, tensor):
        """Apply sliding window offsets to receptive field tensor.

        Args:
            tensor: torch.Tensor of shape (lut_rank, num_kernels, sample_size, 4)
                where last dim is (h, w, d, c).

        Returns:
            out: torch.Tensor of shape (lut_rank, num_kernels, num_positions, sample_size, 4),
                with the sliding-window offsets applied.
        """
        h, w, d = self.in_dim
        h_k, w_k, d_k = self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        d_padded = d + 2 * self.padding

        assert (h_k <= h_padded and w_k <= w_padded and d_k <= d_padded), (
            f"Receptive field size ({h_k}, {w_k}, {d_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}, {d_padded}) after padding."
        )

        # Sliding positions
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)
        d_starts = torch.arange(0, d_padded - d_k + 1, self.stride, device=self.device)

        # Meshgrid for all receptive-field start positions
        h_grid, w_grid, d_grid = torch.meshgrid(h_starts, w_starts, d_starts, indexing="ij")
        h_offsets = h_grid.flatten()
        w_offsets = w_grid.flatten()
        d_offsets = d_grid.flatten()
        num_positions = h_offsets.numel()

        # tensor: (L, K, S, 4) → (K, L, S, 4)
        pairs_all = tensor.permute(1, 0, 2, 3)
        K, L, S, _ = pairs_all.shape

        # Split h, w, d, c coordinates: (K, L, S)
        h_base = pairs_all[..., 0]
        w_base = pairs_all[..., 1]
        d_base = pairs_all[..., 2]
        c_base = pairs_all[..., 3]

        # Add sliding-window offsets (broadcasted) → (K, P, L, S)
        h_idx = h_base.unsqueeze(1) + h_offsets.view(1, num_positions, 1, 1)
        w_idx = w_base.unsqueeze(1) + w_offsets.view(1, num_positions, 1, 1)
        d_idx = d_base.unsqueeze(1) + d_offsets.view(1, num_positions, 1, 1)
        c_idx = c_base.unsqueeze(1).expand(-1, num_positions, -1, -1)

        # Combine back into indices: (K, P, L, S, 4)
        all_indices = torch.stack([h_idx, w_idx, d_idx, c_idx], dim=-1)

        # Reorder so first axis is L: (L, K, P, S, 4)
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

    def get_lut_ids(self):
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
                luts, ids = self.parametrization.get_lut_ids(w)
                level_ids.append(ids)
                level_luts.append(luts)
            tree_ids.append(level_ids)
            tree_luts.append(level_luts)
        return tree_luts, tree_ids


class OrPooling(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0):
        super(OrPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool the max value in the kernel."""
        if x.dim() == 4:
            x = torch.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        elif x.dim() == 5:
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
