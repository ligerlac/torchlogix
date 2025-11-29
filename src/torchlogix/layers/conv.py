from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple
from torch.nn.functional import gumbel_softmax

from .initialization import initialize_weights_raw, initialize_weights_walsh

from ..functional import bin_op_cnn, bin_op_cnn_walsh, gumbel_sigmoid, softmax, sigmoid, walsh_cnn, walsh_basis_hard_cnn_first_level, walsh_basis_hard_cnn_deep_level, walsh_hadamard_transform

##########################################################################


def setup_cnn_cls(parametrization: str) -> torch.nn.Module:
    """Factory function to select the appropriate LogicConv2d subclass.

    Args:
        parametrization: String specifying the LUT parametrization method.
            Supported values are:
            - ``"raw"``: Raw LUT-space logits.
            - ``"walsh"``: Walsh–Hadamard basis coefficients.

    Returns:
        The corresponding LogicDense subclass.
    """
    if parametrization == "raw":
        return LogicConv2d
    elif parametrization == "walsh":
        return LogicConv2dWalsh
    else:
        raise ValueError(f"Unsupported parametrization: {parametrization}")
    

class LogicConv2d(nn.Module):
    """2D convolutional layer with differentiable logic operations.

    This layer implements a 2D convolution where each output location is
    computed by evaluating a learned logic tree over a receptive field.
    Instead of linear filters, it uses a binary tree of differentiable
    logic operations (LUTs) applied to selected positions in the receptive
    field, per kernel and per spatial location.

    Args:
        in_dim: Input spatial dimensions ``(height, width)``.
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
        temperature: Temperature used by (Gumbel-)Softmax sampling of LUT
            weights at each tree node.
        forward_sampling: Sampling strategy for LUT weights. One of:
            - ``"soft"``: Softmax with continuous relaxation.
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax relaxation.
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax.
        residual_init_param: Scalar controlling the strength of the
            residual-style initialization when ``weight_init == "residual"``.
        n_inputs: Arity of each logic gate (number of Boolean inputs per
            node). Typically 2 for binary trees.
        arbitrary_basis: If ``True``, allows more general bases for the LUT
            parametrization (e.g., Walsh), rather than a fixed hard-coded
            basis.
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
        connections: str = "random",  # or 'random-unique'
        weight_init: str = "residual",  # "residual" or "random"
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft", # or "hard", "gumbel_soft", or "gumbel_hard"
        residual_init_param: float = 1.0,
        n_inputs: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    ):
        super().__init__()
        self.forward_sampling = forward_sampling
        self.n_inputs = n_inputs
        self.n_exp = 1 << n_inputs
        self.residual_init_param = residual_init_param
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})."
        )
        self.tree_depth = tree_depth
        self.weight_init = weight_init
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
            self.kernels = self._get_random_receptive_field_tensor()
        elif connections == "random-unique":
            self.kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        # list of tensors, one tensor for each tree depth
        self.indices = self._get_indices_from_kernel_tensor(self.kernels)
        self.temperature = temperature
        self.arbitrary_basis = arbitrary_basis
        self.tree_weights, self.forward_sampling_func = self._init_weights()

    def _init_weights(self):
        tree_weights = torch.nn.ModuleList()
        for i in reversed(range(self.tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(self.n_inputs**i):  # Iterate over nodes at this level
                weights = initialize_weights_raw(self.weight_init, self.num_kernels, 
                                                 self.n_inputs, self.residual_init_param, self.device)
                level_weights.append(torch.nn.Parameter(weights))
            tree_weights.append(level_weights)
        forward_sampling_func = {
            "soft": lambda w: softmax(w, tau=self.temperature, hard=False),
            "hard": lambda w: softmax(w, tau=self.temperature, hard=True),
            "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]
        return tree_weights, forward_sampling_func

    def forward(self, x):
        """Applies the logic convolution to the input.

        The forward pass proceeds as follows:

        1. Optionally pad the input spatially.
        2. Select all receptive-field positions for the first tree level using
           precomputed index tensors.
        3. For each tree level:
            a. Sample or select LUT weights for all nodes at that level.
            b. Apply binary logic operations (via :func:`bin_op_cnn`) to the
               child activations, reducing them up the tree.
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
        # first level tree indices
        ind_h, ind_w, ind_c = self.indices[0][...,0], self.indices[0][...,1], self.indices[0][...,2]
        x = x[:, ind_c, ind_h, ind_w]

        level_weights = torch.stack(
            [self.forward_sampling_func(w) for w in self.tree_weights[0]], dim=0
        )
        if not self.training:
            level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 1 << self.n_exp).to(
                torch.float32
            )
        x = bin_op_cnn(x[:, 0], x[:, 1], level_weights)

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            level_weights = torch.stack(
                [self.forward_sampling_func(w) for w in self.tree_weights[level]], dim=0
            )
            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 1 << self.n_exp).to(
                    torch.float32
                )
            x = bin_op_cnn(x[..., 0, :], x[..., 1, :], level_weights)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size) // self.stride + 1

        x = x.view(
            x.shape[0],
            x.shape[1],
            reshape_h,
            reshape_w,
        )

        return x

    def _get_random_receptive_field_tensor(self):
        """Generate random index tensor within the receptive field for each kernel.
        May contain self connections and duplicate connections.

        Returns:
            indices: (num_kernels, n_inputs, sample_size, 3)
                    where the last dim is (h, w, c)
        """
        c = self.channels
        h_k = self.receptive_field_size
        w_k = self.receptive_field_size
        sample_size = self.n_inputs ** self.tree_depth

        size = (self.num_kernels, self.n_inputs, sample_size)

        h_indices = torch.randint(0, h_k, size, device=self.device)
        w_indices = torch.randint(0, w_k, size, device=self.device)
        c_indices = torch.randint(0, c,   size, device=self.device)

        # shape: (num_kernels, n_inputs, sample_size, 3)
        indices = torch.stack((h_indices, w_indices, c_indices), dim=-1)

        return indices.transpose(0, 1)

    def _get_random_unique_receptive_field_tensor(self):
        """
        Generate random unique index tensor within the receptive field for each kernel.
        - No self-connections inside a tuple (all positions distinct).
        - No duplicate tuples within each kernel (unordered combinations).

        Returns:
            coords: tensor of shape (tuple_size, num_kernels, sample_size, 3)
                    coords[t, k, s] is the (h, w, c) of the t-th element of the s-th tuple
                    for kernel k.
        """
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = self.n_inputs ** self.tree_depth
        device = self.device

        # ---- 1) All RF positions as (h, w, c) ----
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        c_rf = torch.arange(0, c,   device=device)

        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack(
            [h_rf_grid.flatten(), w_rf_grid.flatten(), c_rf_grid.flatten()],
            dim=1,  # (num_positions, 3)
        )

        num_positions = h_k * w_k * c

        # ---- 2) All unique unordered index tuples of size `tuple_size` ----
        # Each row in `comb` is a tuple (i0, i1, ..., i_{tuple_size-1}), i0 < i1 < ...,
        # with all indices in [0, num_positions).
        positions_1d = torch.arange(num_positions, device=device)
        comb = torch.combinations(positions_1d, r=self.n_inputs, with_replacement=False)  # (T, tuple_size)
        total_tuples = comb.shape[0]

        if sample_size > total_tuples:
            raise ValueError(
                f"Not enough unique {self.n_inputs}-tuples: need {sample_size}, have {total_tuples}"
            )

        K = self.num_kernels

        # ---- 3) Sample `sample_size` tuples per kernel, without replacement ----
        selected_idx = torch.multinomial(
            torch.ones(K, total_tuples, device=device),
            sample_size,
            replacement=False
        )

        # Selected tuples of indices into all_positions: (K, sample_size, tuple_size)
        tuple_indices = comb[selected_idx]

        # ---- 4) Map to (h, w, c) coordinates ----
        coords = all_positions[tuple_indices]  # (K, sample_size, tuple_size, 3)

        # ---- 5) Put tuple axis first, to mirror your old (A, B) output style ----
        # Old pairs: (A, B) each (K, S, 3)
        # New tuples: coords[t] is like "A", "B", "C", ...:
        coords = coords.permute(2, 0, 1, 3)    # (tuple_size, K, sample_size, 3)

        return coords
    
    def _apply_sliding_window_tensor(self, tensor):
        """
        tensor: torch.Tensor of shape (n_inputs, num_kernels, sample_size, 3)
            where last dim is (h, w, c).

        Returns:
            out: torch.Tensor of shape (n_inputs, num_kernels, num_positions, sample_size, 3),
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
        h_offsets = h_grid.flatten()        # (P,)
        w_offsets = w_grid.flatten()        # (P,)
        num_positions = h_offsets.numel()   # P

        # tensor: (L, K, S, 3) → (K, L, S, 3) to match old logic
        pairs_all = tensor.permute(1, 0, 2, 3)   # (K, L, S, 3)
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
        indices = [
            self._apply_sliding_window_tensor(tensor)
        ]
        for level in range(self.tree_depth):
            size = self.n_inputs ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.n_inputs).transpose(0, 1)
            indices.append(base)
        return indices
    
    def get_lut_ids(self):
        """Computes the most probable LUT and its ID for each neuron by choosing maximum weight over all
        possible Boolean functions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2 ** n_inputs)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each
                  entry is the integer ID of the corresponding LUT, obtained by
                  interpreting its truth table as a binary number.
        """
        tree_ids = []
        tree_luts = []
        for level in range(self.tree_depth + 1):
            level_ids = []
            level_luts = []
            for w in self.tree_weights[level]:
                ids = w.argmax(axis=1)
                luts = ((ids.unsqueeze(-1) >> torch.arange(1 << self.n_inputs, device=ids.device)) & 1).flip(1)
                level_ids.append(ids)
                level_luts.append(luts)
            tree_ids.append(level_ids)
            tree_luts.append(level_luts)
        return tree_luts, tree_ids
    

class LogicConv2dWalsh(LogicConv2d):
    """2D convolutional layer using Walsh–Hadamard–parametrized logic operations.

    This class extends :class:`LogicConv2d` by replacing the raw-LUT
    parametrization with a Walsh–Hadamard basis (or optionally an arbitrary
    user-defined basis). Each logic gate in the convolution's binary tree is
    parameterized by Walsh coefficients rather than direct truth-table logits.

    This provides a smoother optimization landscape and better structure
    for Boolean functions, especially for parity-like computations.

    This constructor calls :class:`LogicConv2d` first to set up the
    receptive-field connection structure and binary-tree indexing. The
    Walsh-specific weight initialization is then performed by overriding
    :meth:`_init_weights`.

    Args:
        in_dim: Input spatial dimensions ``(height, width)``.
        device: Device for parameter storage (e.g. ``"cpu"``, ``"cuda"``).
        grad_factor: Gradient scaling factor applied inside the logic tree.
        channels: Number of input channels.
        num_kernels: Number of output kernels (analogous to output channels).
        tree_depth: Depth of the binary logic tree.
        receptive_field_size: Spatial size of the receptive field (square).
        connections: Receptive-field sampling mode:
            - ``"random"``: Random RF positions.
            - ``"random-unique"``: Non-overlapping unique positions.
        weight_init: Weight initialization method for Walsh coefficients.
            One of ``"residual"`` or ``"random"``.
        stride: Convolution stride.
        padding: Zero-padding on all sides before RF extraction.
        temperature: Temperature for (Gumbel-)sigmoid sampling.
        forward_sampling: Sampling method for LUT outputs:
            - ``"soft"``, ``"hard"``, ``"gumbel_soft"``, ``"gumbel_hard"``.
        residual_init_param: Scalar controlling the “residual” Walsh init.
        n_inputs: Arity of each logic node (usually 2).
        arbitrary_basis: If ``True``, allows arbitrary basis functions
            rather than the built-in fast Walsh basis.
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
        connections: str = "random",  # or 'random-unique'
        weight_init: str = "residual",  # "residual" or "random"
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        forward_sampling: str = "soft", # or "hard", "gumbel_soft", or "gumbel_hard"
        residual_init_param: float = 1.0,
        n_inputs: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    ):
        super().__init__(
            in_dim=in_dim,
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
            residual_init_param=residual_init_param,
            n_inputs=n_inputs,
            arbitrary_basis=arbitrary_basis,
        )


    def _init_weights(self):
        tree_weights = torch.nn.ModuleList()
        for i in reversed(range(self.tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(self.n_inputs**i):  # Iterate over nodes at this level
                weights = initialize_weights_walsh(self.weight_init, self.num_kernels, self.n_inputs, self.residual_init_param, self.device)
                level_weights.append(torch.nn.Parameter(weights))
            tree_weights.append(level_weights)
        forward_sampling_func = {
            "soft": lambda w: sigmoid(w, tau=self.temperature, hard=False),
            "hard": lambda w: sigmoid(w, tau=self.temperature, hard=True),
            "gumbel_soft": lambda w: gumbel_sigmoid(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_sigmoid(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]
        return tree_weights, forward_sampling_func
    
    def forward(self, x):
        """Applies the Walsh-based logic convolution to the input.

        The computation mirrors :meth:`LogicConv2d.forward`, but replaces raw
        truth-table operations with fast Walsh–Hadamard-domain operations.

        Workflow:
            1. Pad the input if needed.
            2. Extract receptive-field slices for the first tree level.
            3. Compute Walsh basis activations using specialized kernels.
            4. For each tree node:
                - Apply Walsh coefficients to the basis.
                - Aggregate the results upward in the tree.
            5. Reshape the output into ``(batch, num_kernels, H_out, W_out)``.

        Args:
            x: Input tensor of shape ``(batch_size, channels, height, width)``.

        Returns:
            Tensor of shape ``(batch_size, num_kernels, out_height, out_width)``.
        """
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding, 0, 0),
                mode="constant",
                value=0
            )
        # first level tree indices
        ind_h, ind_w, ind_c = self.indices[0][...,0], self.indices[0][...,1], self.indices[0][...,2]
        x = x[:, ind_c, ind_h, ind_w]

        level_weights = torch.stack([w for w in self.tree_weights[0]], dim=0)
        if not self.arbitrary_basis:
            basis = walsh_basis_hard_cnn_first_level(x, self.n_inputs)
        else:
            raise NotImplementedError("Arbitrary basis not implemented yet")
        x = walsh_cnn(basis, level_weights)
        if self.training:
            x = self.forward_sampling_func(-x)
        else:
            x = (-x > 0).to(torch.float32)

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            level_weights = torch.stack([w for w in self.tree_weights[level]], dim=0)
            if not self.arbitrary_basis:
                basis = walsh_basis_hard_cnn_deep_level(x, self.n_inputs)
            else:
                raise NotImplementedError("Arbitrary basis not implemented yet")
            x = walsh_cnn(basis, level_weights)
            if self.training:
                x = self.forward_sampling_func(-x)
            else:
                x = (-x > 0).to(torch.float32)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size) // self.stride + 1

        x = x.view(
            x.shape[0],
            x.shape[1],
            reshape_h,
            reshape_w,
        )

        return x
    
    def get_lut_ids(self):
        """
        Computes most-probable LUT for each learned set of weights.
        Returns tuple with most probable LUTs and their IDs.
        """
        tree_ids = []
        tree_luts = []
        for level in range(self.tree_depth + 1):
            level_ids = []
            level_luts = []
            for w in self.tree_weights[level]:
                luts = walsh_hadamard_transform(w, self.n_inputs)
                luts = luts < 0
                ids = 2 ** torch.arange((1 << self.n_inputs) - 1, -1, -1, device=luts.device)
                ids = (luts * ids.unsqueeze(0)).sum(dim=1)
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
        padding: Zero-padding applied symmetrically to height and width
            before selecting receptive fields.
        temperature: Temperature used by (Gumbel-)Softmax sampling of LUT
            weights at each tree node.
        forward_sampling: Sampling strategy for LUT weights. One of:
            - ``"soft"``: Softmax with continuous relaxation.
            - ``"hard"``: Straight-through hard selection.
            - ``"gumbel_soft"``: Gumbel-Softmax relaxation.
            - ``"gumbel_hard"``: Straight-through Gumbel-Softmax.
        residual_init_param: Scalar controlling the strength of the
            residual-style initialization when ``weight_init == "residual"``.
        n_inputs: Arity of each logic gate (number of Boolean inputs per
            node). Typically 2 for binary trees.
        arbitrary_basis: If ``True``, allows more general bases for the LUT
            parametrization (e.g., Walsh), rather than a fixed hard-coded
            basis.
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
        connections: str = "random",  # or 'random-unique'
        weight_init: str = "residual",  # "residual" or "random"
        stride: int = 1,
        padding: int = None,
        temperature: float = 1.0,
        forward_sampling: str = "soft", # or "hard", "gumbel_soft", or "gumbel_hard"
        residual_init_param: float = 1.0,
        n_inputs: int = 2,
        arbitrary_basis: bool = False  # Whether to use hard-coded basis
    
    ):
        super().__init__()
        self.receptive_field_size = _triple(receptive_field_size)
        self.forward_sampling = forward_sampling
        self.n_inputs = n_inputs
        self.n_exp = 1 << n_inputs
        self.residual_init_param = residual_init_param
        self.weight_init = weight_init
        self.in_dim = _triple(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.connections = connections
        self.temperature = temperature
        self.arbitrary_basis = arbitrary_basis
        assert (
            (stride <= self.receptive_field_size[0]) and
            (stride <= self.receptive_field_size[1]) and
            (stride <= self.receptive_field_size[2])), (
                f"Stride ({stride}) cannot be larger than receptive field size "
                f"({receptive_field_size})"
            )
        if connections == "random":
            self.kernels = self._get_random_receptive_field_tensor()
        elif connections == "random-unique":
            self.kernels= self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        self.indices = self._get_indices_from_kernel_tensor(self.kernels)
        self.tree_weights, self.forward_sampling_func = self._init_weights()

    
    def _init_weights(self):
        tree_weights = torch.nn.ModuleList()
        for i in reversed(range(self.tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(self.n_inputs**i):  # Iterate over nodes at this level
                weights = initialize_weights_raw(self.weight_init, self.num_kernels, 
                                                 self.n_inputs, self.residual_init_param, self.device)
                level_weights.append(torch.nn.Parameter(weights))
            tree_weights.append(level_weights)
        forward_sampling_func = {
            "soft": lambda w: softmax(w, tau=self.temperature, hard=False),
            "hard": lambda w: softmax(w, tau=self.temperature, hard=True),
            "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]
        return tree_weights, forward_sampling_func
    
    def forward(self, x):
        """Applies the logic convolution to the input.

        The forward pass proceeds as follows:

        1. Optionally pad the input spatially.
        2. Select all receptive-field positions for the first tree level using
           precomputed index tensors.
        3. For each tree level:
            a. Sample or select LUT weights for all nodes at that level.
            b. Apply binary logic operations (via :func:`bin_op_cnn`) to the
               child activations, reducing them up the tree.
        4. Reshape the final per-kernel outputs into a 5D tensor of shape
           ``(batch_size, num_kernels, out_height, out_width, out_depth)``.

        Args:
            x: Input tensor of shape ``(batch_size, channels, height, width, depth)``.

        Returns:
            Tensor of shape ``(batch_size, num_kernels, out_height, out_width, out_depth)``,
            where ``out_height``, ``out_width``, and ``out_depth`` are determined by the
            convolution parameters:

            * ``out_height = (in_height + 2 * padding - receptive_field_size) // stride + 1``
            * ``out_width  = (in_width  + 2 * padding - receptive_field_size) // stride + 1``.
            * ``out_depth  = (in_depth  + 2 * padding - receptive_field_size) // stride + 1``.
        """
        ind_h, ind_w, ind_d, ind_c = self.indices[0][...,0], self.indices[0][...,1], self.indices[0][...,2], self.indices[0][...,3]
        x = x[:, ind_c, ind_h, ind_w, ind_d]

        # Process first level
        level_weights = torch.stack(
            [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[0]],
            dim=0,
        )
        if not self.training:
            level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 1 << self.n_exp).to(
                torch.float32
            )

        x = bin_op_cnn(x[:, 0], x[:, 1], level_weights)

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            x = x[..., self.indices[level]]
            level_weights = torch.stack(
                [self.forward_sampling_func(w) for w in self.tree_weights[level]], dim=0,
            )

            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 1 << self.n_exp).to(
                    torch.float32
                )

            x = bin_op_cnn(x[..., 0, :], x[..., 1, :], level_weights)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size[0]) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size[1]) // self.stride + 1
        reshape_d = (self.in_dim[2] + 2*self.padding - self.receptive_field_size[2]) // self.stride + 1

        x = x.view(
            x.shape[0],
            x.shape[1],
            reshape_h,
            reshape_w,
            reshape_d
        )

        return x
    
    def _get_random_receptive_field_tensor(self):
        """Generate random index tensor within the receptive field for each kernel.
        May contain self connections and duplicate connections.

        Returns:
            indices: (num_kernels, n_inputs, sample_size, 4)
                    where the last dim is (h, w, d, c)
        """
        c = self.channels
        h_k = self.receptive_field_size[0]
        w_k = self.receptive_field_size[1]
        d_k = self.receptive_field_size[2]
        sample_size = self.n_inputs ** self.tree_depth

        size = (self.num_kernels, self.n_inputs, sample_size)

        h_indices = torch.randint(0, h_k, size, device=self.device)
        w_indices = torch.randint(0, w_k, size, device=self.device)
        d_indices = torch.randint(0, d_k,   size, device=self.device)
        c_indices = torch.randint(0, c,   size, device=self.device)

        # shape: (num_kernels, n_inputs, sample_size, 4)
        indices = torch.stack((h_indices, w_indices, d_indices, c_indices), dim=-1)
        return indices.transpose(0, 1)
    
    def _get_random_unique_receptive_field_tensor(self):
        """
        Generate random unique index tensor within the receptive field for each kernel.
        - No self-connections inside a tuple (all positions distinct).
        - No duplicate tuples within each kernel (unordered combinations).

        Returns:
            coords: tensor of shape (tuple_size, num_kernels, sample_size, 3)
                    coords[t, k, s] is the (h, w, c) of the t-th element of the s-th tuple
                    for kernel k.
        """
        c, h_k, w_k, d_k = self.channels, *self.receptive_field_size
        sample_size = self.n_inputs ** self.tree_depth
        device = self.device

        # ---- 1) All RF positions as (h, w, c) ----
        h_rf = torch.arange(0, h_k, device=device)
        w_rf = torch.arange(0, w_k, device=device)
        d_rf = torch.arange(0, d_k, device=device)
        c_rf = torch.arange(0, c,   device=device)

        h_rf_grid, w_rf_grid, d_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, d_rf, c_rf, indexing="ij")
        all_positions = torch.stack(
            [h_rf_grid.flatten(), w_rf_grid.flatten(), d_rf_grid.flatten(), c_rf_grid.flatten()],
            dim=1,  # (num_positions, 3)
        )

        num_positions = h_k * w_k * d_k * c

        # ---- 2) All unique unordered index tuples of size `tuple_size` ----
        # Each row in `comb` is a tuple (i0, i1, ..., i_{tuple_size-1}), i0 < i1 < ...,
        # with all indices in [0, num_positions).
        positions_1d = torch.arange(num_positions, device=device)
        comb = torch.combinations(positions_1d, r=self.n_inputs, with_replacement=False)  # (T, tuple_size)
        total_tuples = comb.shape[0]

        if sample_size > total_tuples:
            raise ValueError(
                f"Not enough unique {self.n_inputs}-tuples: need {sample_size}, have {total_tuples}"
            )

        K = self.num_kernels

        # ---- 3) Sample `sample_size` tuples per kernel, without replacement ----
        selected_idx = torch.multinomial(
            torch.ones(K, total_tuples, device=device),
            sample_size,
            replacement=False
        )

        # Selected tuples of indices into all_positions: (K, sample_size, tuple_size)
        tuple_indices = comb[selected_idx]

        # ---- 4) Map to (h, w, c) coordinates ----
        coords = all_positions[tuple_indices]  # (K, sample_size, tuple_size, 4)

        # ---- 5) Put tuple axis first, to mirror your old (A, B) output style ----
        # Old pairs: (A, B) each (K, S, 4)
        # New tuples: coords[t] is like "A", "B", "C", ...:
        coords = coords.permute(2, 0, 1, 3)    # (tuple_size, K, sample_size, 4)
        return coords
    
    def _apply_sliding_window_tensor(self, tensor):
        """
        tensor: torch.Tensor of shape (n_inputs, num_kernels, sample_size, 4)
            where last dim is (h, w, c, d).

        Returns:
            out: torch.Tensor of shape (n_inputs, num_kernels, num_positions, sample_size, 4),
                with the sliding-window offsets applied.
        """
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

        # Sliding positions
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)
        d_starts = torch.arange(0, d_padded - d_k + 1, self.stride, device=self.device)
        # Meshgrid for all receptive-field start positions
        h_grid, w_grid, d_grid = torch.meshgrid(h_starts, w_starts, d_starts, indexing="ij")
        h_offsets = h_grid.flatten()        # (P,)
        w_offsets = w_grid.flatten()        # (P,)
        d_offsets = d_grid.flatten()        # (P,)
        num_positions = h_offsets.numel()   # P

        # tensor: (L, K, S, 3) → (K, L, S, 3) to match old logic
        pairs_all = tensor.permute(1, 0, 2, 3)   # (K, L, S, 3)
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
        indices = [
            self._apply_sliding_window_tensor(tensor)
        ]
        for level in range(self.tree_depth):
            size = self.n_inputs ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.n_inputs).transpose(0, 1)
            indices.append(base)
        return indices
    
    def get_lut_ids(self):
        """Computes the most probable LUT and its ID for each neuron by choosing maximum weight over all
        possible Boolean functions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2 ** n_inputs)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each
                  entry is the integer ID of the corresponding LUT, obtained by
                  interpreting its truth table as a binary number.
        """
        tree_ids = []
        tree_luts = []
        for level in range(self.tree_depth + 1):
            level_ids = []
            level_luts = []
            for w in self.tree_weights[level]:
                ids = w.argmax(axis=1)
                luts = ((ids.unsqueeze(-1) >> torch.arange(1 << self.n_inputs, device=ids.device)) & 1).flip(1)
                level_ids.append(ids)
                level_luts.append(luts)
            tree_ids.append(level_ids)
            tree_luts.append(level_luts)
        return tree_luts, tree_ids


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
