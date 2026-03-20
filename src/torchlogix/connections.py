from typing import Union
from abc import ABC, abstractmethod
import itertools

import torch
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from .functional import softmax, take_tuples, get_combination_indices
    

def setup_connections(
    connections: str,
    structure: str,
    lut_rank: int,
    device: str = None,
    **connections_kwargs
):
    """Factory method to create connection modules."""
    if structure == "dense":
        if connections == "fixed":
            return FixedDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        elif connections == "learnable":
            return LearnableDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    elif structure == "conv":
        if connections == "fixed":
            return FixedConvConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    else:
        raise ValueError(f"Unknown structure method: {structure}")
    

class Connections(torch.nn.Module, ABC):
    """Abstract base class for connection strategies."""
    def __init__(
            self,
            lut_rank=2,
            device=None,
            init_method="random",
            **kwargs
        ):
        super().__init__()
        self.lut_rank = lut_rank
        self.device = device
        self.init_method = init_method

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _init_connections(self):
        pass

    def update_temperature(self, temperature: float):
        pass


class FixedDenseConnections(Connections):
    """Fixed dense connections implementation.
    Each neuron connects to ``lut_rank`` input features chosen from the input dimension.
    The connections are fixed after initialization.
    
    Args:
        in_dim: Input feature dimension.
        out_dim: Number of neurons (output dimension).
        lut_rank: Number of input features each neuron connects to.
        device: Device to store the connection indices tensor.
        init_method: Method to initialize connections. Options are:
            - "random": Randomly select input features (with replacement).
            - "random-unique": Randomly select unique input features (without replacement).
    """
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            lut_rank=2, 
            device=None,
            init_method="random",
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.indices = self._init_connections()

    def _init_connections(self):
        """Constructs possible input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features chosen out of ``lut_rank * num_candidates``
        possibilities. This function returns a tensor encoding which input indices are connected 
        to which neuron.

        Returns:
            A tensor of shape ``(num_candidates, lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.lut_rank, (
            f"Cannot have num_candidates * lut_rank > in_dim "
            f"({self.lut_rank} > {self.in_dim})"
        )
        assert self.out_dim * self.lut_rank >= self.in_dim, (
                f"Need out_dim * lut_rank >= in_dim to cover all inputs "
                f"({self.out_dim} * {self.lut_rank} < {self.in_dim})."
                )

        if self.init_method == "random":
            # With this method both inputs can stem from the same input feature
            c = torch.randperm(self.lut_rank * self.out_dim, 
                               device=self.device) % self.in_dim
            c = c.reshape(self.lut_rank, self.out_dim)
        elif self.init_method == "random-unique":
            c = get_random_unique_connections(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                n=self.lut_rank
            )
        else:
            raise ValueError(self.init_method)
        c = c.contiguous().to(torch.int64).to(self.device)
        return c
    
    def forward(self, x):
        return x[:, self.indices]
    

class LearnableConnectionFunction(torch.autograd.Function):
    """Autograd function for learnable connections.
    Implements the forward and backward pass for learnable connections
    using Gumbel-Softmax for differentiable sampling.
    """
    @staticmethod
    def forward(ctx, x, weights, tau, gumbel, indices):
        if gumbel:
            u = torch.rand_like(weights)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        else:
            g = torch.zeros_like(weights)
        connections = (weights + g).argmax(dim=0)
        l = torch.arange(weights.shape[1], device=x.device).unsqueeze(1)
        o = torch.arange(weights.shape[2], device=x.device).unsqueeze(0)
        output = x[:, indices[connections, l, o]]
        ctx.save_for_backward(x, weights, tau, g, indices)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau, g, indices = ctx.saved_tensors
        # compute gradient w.r.t. to learnable weights
        weights_grad = torch.einsum("bclo,blo->clo", 2*x[:,indices]-1, output_grad)
        # compute gradient w.r.t. to input with sparse scatter_add method
        input_grad = torch.zeros_like(x)
        temp = softmax((weights + g)/tau, dim=0).unsqueeze(0) * output_grad.unsqueeze(1)
        input_grad.scatter_add_(dim=1, 
                                index=indices.expand(x.shape[0], -1, -1, -1).reshape(x.shape[0], -1),
                                src=temp.reshape(x.shape[0], -1))
        return input_grad, weights_grad, None, None, None
    

class LearnableDenseConnections(Connections):
    """Learnable dense connections implementation.
    Each neuron connects to ``lut_rank`` input features chosen from a set of candidates.
    The connections are learnable parameters optimized during training.

    Args:
        in_dim: Input feature dimension.
        out_dim: Number of neurons (output dimension).
        lut_rank: Number of input features each neuron connects to.
        temperature: Temperature parameter for Gumbel-Softmax.
        num_candidates: Number of candidate input features per neuron.
            If -1, all input features are candidates.
        gumbel: Whether to use Gumbel noise for sampling.
        device: Device to store the connection indices tensor.
        init_method: Method to initialize connections. Options are:
            - "random": Randomly select input features (with replacement).
            - "random-unique": Randomly select unique input features (without replacement).
    """
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            lut_rank=2, 
            temperature=0.001,
            num_candidates=-1, 
            gumbel=False,
            device=None,
            init_method="random",
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )
        self.temperature = temperature
        self.num_candidates = num_candidates
        self.lut_rank = lut_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.gumbel = gumbel
        if num_candidates == -1:
            num_candidates = in_dim
            self.num_candidates = num_candidates
            self.indices = torch.arange(in_dim, device=self.device).view(
                in_dim, 1, 1).expand(in_dim, lut_rank, out_dim)
        else:
            assert num_candidates > 0, "num_candidates must be bigger than 0"
            self.num_candidates = num_candidates
            self.indices = self._init_connections()
        self.weights = torch.nn.Parameter(torch.rand(
            num_candidates, lut_rank, out_dim, dtype=torch.float32), requires_grad=True)
        
    def update_temperature(self, temperature: float):
        self.temperature = temperature
        
    def forward(self, x):
        return LearnableConnectionFunction.apply(x, self.weights, torch.tensor(self.temperature), 
                                                 self.gumbel, self.indices)
    
    def _init_connections(self):
        """Constructs possible input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features chosen out of ``lut_rank * num_candidates``
        possibilities. This function returns a tensor encoding which input indices are connected 
        to which neuron.

        Returns:
            A tensor of shape ``(num_candidates, lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.num_candidates * self.lut_rank, (
            f"Cannot have num_candidates * lut_rank > in_dim "
            f"({self.num_candidates * self.lut_rank} > {self.in_dim})"
        )
        assert self.out_dim * self.lut_rank >= self.in_dim, (
                f"Need out_dim * lut_rank >= in_dim to cover all inputs "
                f"({self.out_dim} * {self.lut_rank} < {self.in_dim})."
                )

        if self.init_method == "random":
            # With this method both inputs can stem from the same input feature
            c = torch.randperm(self.lut_rank * self.out_dim * self.num_candidates, 
                               device=self.device) % self.in_dim
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
        elif self.init_method == "random-unique":
            c = get_random_unique_connections(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                n=self.lut_rank*self.num_candidates
            )
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
        else:
            raise ValueError(self.connections)
        c = c.contiguous().to(torch.int64).to(self.device)
        return c
    

def get_random_unique_connections(in_dim, out_dim, n):
    # Feasibility check
    n_max = int(in_dim * (in_dim // (n - 1) - 1) / 2)
    assert out_dim <= n_max, (
        "The number of neurons ({}) must not be greater than the number of pair-wise combinations "
        "of the inputs ({})".format(out_dim, n_max)
    )
    x = torch.arange(in_dim)
    c = take_tuples(x, tuple_size=n, stride_within=1)
    offset = 2
    while c.shape[-1] < out_dim:
        c_ = take_tuples(x, tuple_size=n, stride_within=offset)
        c = torch.cat([c, c_], dim=-1)
        offset += 1
    c = c[:, :out_dim]
    perm_out = torch.randperm(out_dim)
    perm_in = torch.randperm(in_dim)
    c = c[:, perm_out]
    c = perm_in[c]
    return c


class FixedConvConnections(Connections):
    """Fixed convolutional connections implementation.
    Each convolutional kernel connects to input features within its receptive field.
    The connections are fixed after initialization."""
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
            lut_rank=2, 
            device=None,
            init_method="random",  # | "random-unique"
            channel_group_size: int = None,
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
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
        self.channel_group_size = channel_group_size
        if channel_group_size is not None:
            assert channels > channel_group_size, (
                "channel_group_size must be smaller than the number of channels"
            )
        self.indices = self._init_connections()
        
        
    def _init_connections(self):
        # Setup connections
        if self.init_method == "random":
            kernels = self._get_random_receptive_field_tensor()
        elif self.init_method == "random-unique":
            kernels = self._get_random_unique_receptive_field_tensor()
        else:
            raise ValueError(f"Unknown connections type: {self.init_method}")
        # Build tree indices
        return self._get_indices_from_kernel_tensor(kernels)


    def _get_random_receptive_field_tensor(self):
        """
        Random sampling (with replacement).

        Returns:
            coords: (lut_rank, num_kernels, sample_size, 3)
        """

        c = self.channels
        g = self.channel_group_size
        device = self.device

        sample_size = self.lut_rank ** (self.tree_depth - 1)
        total_inputs = self.lut_rank * sample_size

        # ---------------------------
        # Precompute spatial grid
        # ---------------------------
        rf_axes = [
            torch.arange(0, dim, device=device)
            for dim in self.receptive_field_size
        ]

        spatial_grid = torch.meshgrid(*rf_axes, indexing="ij")
        spatial_positions = torch.stack(
            [grid.flatten() for grid in spatial_grid], dim=1
        )
        num_spatial = spatial_positions.shape[0]

        # ---------------------------
        # Channel group setup
        # ---------------------------
        if g is None:
            starts = None
        else:
            starts = torch.arange(0, c - g + 1, device=device)
            num_groups = starts.numel()

        coords_per_kernel = []

        for k in range(self.num_kernels):

            if g is None:
                c_rf = torch.arange(0, c, device=device)

                # full 3D position space - channel first, then spatial (c, h, w)
                grid = torch.meshgrid(c_rf, *rf_axes, indexing="ij")
                all_positions = torch.stack(
                    [grid_i.flatten() for grid_i in grid], dim=1
                )
                num_positions = all_positions.shape[0]

                idx = torch.randint(
                    0, num_positions,
                    (sample_size, self.lut_rank),
                    device=device,
                )

                coords_k = all_positions[idx]

            else:
                start = starts[k % num_groups]
                c_rf = start + torch.arange(g, device=device)

                if total_inputs % g != 0:
                    raise ValueError(
                        f"Cannot evenly distribute {total_inputs} across {g} channels."
                    )

                inputs_per_channel = total_inputs // g
                channel_chunks = []

                for channel in c_rf:
                    idx = torch.randint(
                        0, num_spatial,
                        (inputs_per_channel,),
                        device=device,
                    )

                    chosen = spatial_positions[idx]

                    ch_col = torch.full(
                        (inputs_per_channel, 1),
                        channel,
                        device=device,
                    )

                    channel_chunks.append(
                        torch.cat([ch_col, chosen], dim=1)
                    )

                coords_k = torch.cat(channel_chunks, dim=0)

                perm = torch.randperm(total_inputs, device=device)
                coords_k = coords_k[perm]

                coords_k = coords_k.view(sample_size, self.lut_rank, 3)

            coords_per_kernel.append(coords_k)

        coords = torch.stack(coords_per_kernel, dim=0)
        coords = coords.permute(2, 0, 1, 3)

        return coords
    
    
    def _get_random_unique_receptive_field_tensor(self):
        """
        Random unique sampling (without replacement across tuples).

        Returns:
            coords: (lut_rank, num_kernels, sample_size, 3)
        """

        c = self.channels
        g = self.channel_group_size
        device = self.device

        sample_size = self.lut_rank ** (self.tree_depth - 1)
        total_inputs = self.lut_rank * sample_size

        # ---------------------------
        # Precompute spatial grid
        # ---------------------------
        rf_axes = [
            torch.arange(0, dim, device=device)
            for dim in self.receptive_field_size
        ]

        spatial_grid = torch.meshgrid(*rf_axes, indexing="ij")
        spatial_positions = torch.stack(
            [grid.flatten() for grid in spatial_grid], dim=1
        )
        num_spatial = spatial_positions.shape[0]

        # ---------------------------
        # Channel group setup
        # ---------------------------
        if g is None:
            starts = None
        else:
            starts = torch.arange(0, c - g + 1, device=device)
            num_groups = starts.numel()

        coords_per_kernel = []

        for k in range(self.num_kernels):

            if g is None:
                c_rf = torch.arange(0, c, device=device)

                # Channel first, then spatial (c, h, w)
                grid = torch.meshgrid(c_rf, *rf_axes, indexing="ij")
                all_positions = torch.stack(
                    [grid_i.flatten() for grid_i in grid], dim=1
                )

                # all unique lut_rank combinations
                all_indices = list(
                    itertools.combinations(
                        range(all_positions.shape[0]),
                        self.lut_rank,
                    )
                )

                if len(all_indices) < sample_size:
                    raise ValueError("Not enough unique combinations.")

                chosen = torch.randperm(
                    len(all_indices),
                    device=device
                )[:sample_size]

                selected = [
                    torch.tensor(all_indices[i], device=device)
                    for i in chosen
                ]

                idx = torch.stack(selected, dim=0)
                coords_k = all_positions[idx]

            else:
                start = starts[k % num_groups]
                c_rf = start + torch.arange(g, device=device)

                if total_inputs % g != 0:
                    raise ValueError(
                        f"Cannot evenly distribute {total_inputs} across {g} channels."
                    )

                inputs_per_channel = total_inputs // g

                if num_spatial < inputs_per_channel:
                    raise ValueError(
                        "Not enough spatial positions for balanced per-channel sampling."
                    )

                channel_chunks = []

                for channel in c_rf:
                    idx = torch.randperm(
                        num_spatial,
                        device=device
                    )[:inputs_per_channel]

                    chosen = spatial_positions[idx]

                    ch_col = torch.full(
                        (inputs_per_channel, 1),
                        channel,
                        device=device,
                    )

                    channel_chunks.append(
                        torch.cat([ch_col, chosen], dim=1)
                    )

                coords_k = torch.cat(channel_chunks, dim=0)

                perm = torch.randperm(total_inputs, device=device)
                coords_k = coords_k[perm]

                coords_k = coords_k.view(sample_size, self.lut_rank, 3)

            coords_per_kernel.append(coords_k)

        coords = torch.stack(coords_per_kernel, dim=0)
        coords = coords.permute(2, 0, 1, 3)

        return coords


    def _apply_sliding_window_tensor(self, tensor):
        """
        Convert (c,h,w) coordinates into flat indices and apply sliding window.

        Args:
            tensor: (L, K, S, 3)  # (lut_rank, kernels, samples, coords)

        Returns:
            indices: (P, L*S, K)  # flat indices for x.view(B, -1)[..., indices]
                Shape allows: x_flat[..., indices] -> (B, P, L*S, K)
        """
        device = self.device

        # ---------------------------
        # Input geometry
        # ---------------------------
        padded = [in_dim + 2 * self.padding for in_dim in self.in_dim]

        # ---------------------------
        # Sliding window positions
        # ---------------------------
        starts = [
            torch.arange(0, p - r + 1, self.stride, device=device)
            for p, r in zip(padded, self.receptive_field_size)
        ]

        # Check if there are any valid positions
        if any(start.numel() == 0 for start in starts):
            # No valid convolution positions - should raise assertion error
            raise AssertionError(
                f"Receptive field size {self.receptive_field_size} is larger than "
                f"padded input dimensions {padded}"
            )

        grid = torch.meshgrid(*starts, indexing="ij")
        offsets = [g.flatten() for g in grid]
        P = offsets[0].numel()

        # ---------------------------
        # Reorder tensor: (K, L, S, 3)
        # ---------------------------
        tensor = tensor.permute(1, 0, 2, 3)

        c = tensor[..., 0]   # (K, L, S)
        spatial = tensor[..., 1:]  # (K, L, S, D)

        # ---------------------------
        # Apply offsets: expand to (K, P, L, S)
        # ---------------------------
        c = c.unsqueeze(1).expand(-1, P, -1, -1)

        spatial_idx = []
        for i, off in enumerate(offsets):
            base = spatial[..., i].unsqueeze(1)  # (K, 1, L, S)
            spatial_idx.append(base + off.view(1, -1, 1, 1))  # (K, P, L, S)

        # ---------------------------
        # Convert to linear indices
        # ---------------------------
        if self.conv_dimension == 2:
            H, W = padded
            h, w = spatial_idx
            linear = c * (H * W) + h * W + w

        else:  # 3D
            D, H, W = padded
            d, h, w = spatial_idx
            linear = c * (D * H * W) + d * (H * W) + h * W + w

        # linear: (K, P, L, S)
        # Rearrange to (P, L*S, K) for efficient indexing
        K, P, L, S = linear.shape
        linear = linear.permute(1, 2, 3, 0)  # (P, L, S, K)
        linear = linear.reshape(P, L * S, K)

        return linear.long()


    def _get_indices_from_kernel_tensor(self, tensor):
        """Build index tensors for all tree levels."""
        # Store coordinate-based tensor for debugging/tests/compilation
        self._coord_tensor = tensor  # (L, K, S, 3)

        # Also build coordinate-based sliding window indices for compilation
        self._coord_indices = self._build_coordinate_indices(tensor)

        # Build flat indices for efficient forward pass
        self._flat_indices = [
            self._apply_sliding_window_tensor(tensor)
        ]
        for level in range(1, self.tree_depth):
            size = self.lut_rank ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.lut_rank).transpose(0, 1)
            self._flat_indices.append(base)

        # Build coordinate-based indices for backward compatibility with tests
        # Format: [(left, right), (left, right), ...] for each tree level
        indices = []
        # Level 0: from _coord_indices (L, P, K, S, ndim) -> [(P, K, S, ndim), (P, K, S, ndim)]
        indices.append([self._coord_indices[0], self._coord_indices[1]])
        # Other levels: split the lut_rank dimension
        for level in range(1, self.tree_depth):
            size = self.lut_rank ** (self.tree_depth - level)
            base = torch.arange(size, device=self.device).view(-1, self.lut_rank).transpose(0, 1)
            indices.append([base[0], base[1]])

        return indices

    def _build_coordinate_indices(self, tensor):
        """Build coordinate-based indices (L, P, K, S, ndim) for compilation."""
        padded = [in_dim + 2 * self.padding for in_dim in self.in_dim]

        starts = [
            torch.arange(0, p - r + 1, self.stride, device=self.device)
            for p, r in zip(padded, self.receptive_field_size)
        ]

        # Check if there are any valid positions
        if any(start.numel() == 0 for start in starts):
            # No valid convolution positions - should raise assertion error
            raise AssertionError(
                f"Receptive field size {self.receptive_field_size} is larger than "
                f"padded input dimensions {padded}"
            )

        grid = torch.meshgrid(*starts, indexing="ij")
        offsets = [g.flatten() for g in grid]
        P = offsets[0].numel()

        # tensor: (L, K, S, ndim)
        L, K, S, ndim = tensor.shape

        # Build indices with sliding window: (L, P, K, S, ndim)
        coord_indices = []
        for lut_idx in range(L):
            lut_positions = []
            for pos_idx in range(P):
                pos_coords = tensor[lut_idx].clone()  # (K, S, ndim)
                # Add spatial offsets
                for dim_idx, offset in enumerate(offsets):
                    pos_coords[..., dim_idx + 1] += offset[pos_idx]
                lut_positions.append(pos_coords)
            coord_indices.append(torch.stack(lut_positions, dim=0))  # (P, K, S, ndim)

        return torch.stack(coord_indices, dim=0)  # (L, P, K, S, ndim)
    

    def forward(self, x, tree_level):
        if tree_level == 0:
            # Flatten input: (B, C, *spatial) -> (B, C*H*W)
            B = x.shape[0]
            x_flat = x.reshape(B, -1)

            # _flat_indices[0]: (P, L*S, K)
            # Indexing: x_flat[..., indices] -> (B, P, L*S, K)
            result = x_flat[..., self._flat_indices[0]]

            # Reshape to (B, L, P, K*S)
            P, LS, K = self._flat_indices[0].shape
            L = self.lut_rank
            S = LS // L
            result = result.reshape(B, P, L, S, K)
            result = result.permute(0, 2, 1, 4, 3)  # (B, L, P, K, S)
            result = result.reshape(B, L, P, K * S)

            return result
        else:
            # For levels > 0: simple indexing
            # Input: (B, S, K, features)
            # _flat_indices: (lut_rank, M)
            result = x[..., self._flat_indices[tree_level]]  # (B, S, K, L, M)

            # Reshape to (B, L, S, K*M)
            B, S, K, L, M = result.shape
            result = result.permute(0, 3, 1, 2, 4).reshape(B, L, S, K * M)

            return result
