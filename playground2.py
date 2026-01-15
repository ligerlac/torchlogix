import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import argparse
import time


def compute_all_logic_ops_vectorized(a, b):
    """Compute all 16 logic operations in a single vectorized operation.
    
    Returns a tensor with shape [..., 16] where the last dimension contains
    all 16 logic operations applied to inputs a and b.
    """
    # Precompute common terms to avoid redundant calculations
    ab = a * b  # AND operation
    a_plus_b = a + b
    a_or_b = a_plus_b - ab  # OR operation
    
    # Stack all 16 operations efficiently using precomputed terms
    ops = torch.stack([
        torch.zeros_like(a),          # 0: 0
        ab,                           # 1: A and B
        a - ab,                       # 2: A and not B
        a,                            # 3: A
        b - ab,                       # 4: B and not A
        b,                            # 5: B
        a_plus_b - 2 * ab,           # 6: A xor B
        a_or_b,                      # 7: A or B
        1 - a_or_b,                  # 8: not(A or B)
        1 - (a_plus_b - 2 * ab),     # 9: not(A xor B)
        1 - b,                       # 10: not B
        1 - b + ab,                  # 11: B implies A
        1 - a,                       # 12: not A  
        1 - a + ab,                  # 13: A implies B
        1 - ab,                      # 14: not(A and B)
        torch.ones_like(a)           # 15: 1
    ], dim=-1)
    
    return ops


def compute_all_logic_ops_vectorized_polish(a, b):
    """Compute all 16 logic operations in a single vectorized operation.
    
    Returns a tensor with shape [..., 16] where the last dimension contains
    all 16 logic operations applied to inputs a and b.
    """
    one = torch.ones_like(a)
    ab = a * b

    # Stack all 16 operations efficiently using precomputed terms
    ops = torch.stack([
        torch.zeros_like(a),         # 0: 0
        ab,                          # 1: A and B
        a - ab,                      # 2: A and not B
        a,                           # 3: A
        b - ab,                      # 4: B and not A
        b,                           # 5: B
        a + b - 2 * ab,              # 6: A xor B
        a + b - ab,                  # 7: A or B
        1 - (a + b - ab),            # 8: not(A or B)
        1 - (a + b - 2 * ab),        # 9: not(A xor B)
        1 - b,                       # 10: not B
        1 - b + ab,                  # 11: B implies A
        1 - a,                       # 12: not A
        1 - a + ab,                  # 13: A implies B
        1 - ab,                      # 14: not(A and B)
        torch.ones_like(a)           # 15: 1
    ], dim=-1)
    
    return ops


def compute_all_logic_ops_vectorized_explicit(a, b):
    """Compute all 16 logic operations in a single vectorized operation.
    
    Returns a tensor with shape [..., 16] where the last dimension contains
    all 16 logic operations applied to inputs a and b.
    """    
    # Stack all 16 operations efficiently using precomputed terms
    ops = torch.stack([
        torch.zeros_like(a),          # 0: 0
        a * b,                           # 1: A and B
        a - a * b,                       # 2: A and not B
        a,                            # 3: A
        b - a * b,                       # 4: B and not A
        b,                            # 5: B
        a + b - 2 * a * b,           # 6: A xor B
        a + b - a * b,                      # 7: A or B
        1 - (a + b - a * b),                  # 8: not(A or B)
        1 - (a + b - 2 * a * b),              # 9: not(A xor B)
        1 - b,                       # 10: not B
        1 - b + a * b,                  # 11: B implies A
        1 - a,                       # 12: not A
        1 - a + a * b,                  # 13: A implies B
        1 - (a * b),                      # 14: not(A and B)
        torch.ones_like(a)           # 15: 1
    ], dim=-1)
    
    return ops


def compute_weighted_logic(a, b, weights):
    """
    a, b: (batch, kernels, positions)
    weights: (kernels, 16)
    returns: (batch, kernels, positions)
    """

    batch_size = weights.shape[0]
    w = weights.view(batch_size, -1, 16)  # (1, kernels, 16)

    output = w[...,0] * 0 \
           + w[...,1] * (a * b) \
           + w[...,2] * (a - a * b) \
           + w[...,3] * a \
           + w[...,4] * (b - a * b) \
           + w[...,5] * b \
           + w[...,6] * (a + b - 2 * a * b) \
           + w[...,7] * (a + b - a * b) \
           + w[...,8] * (1 - (a + b - a * b)) \
           + w[...,9] * (1 - (a + b - 2 * a * b)) \
           + w[...,10] * (1 - b) \
           + w[...,11] * (1 - b + a * b) \
           + w[...,12] * (1 - a) \
           + w[...,13] * (1 - a + a * b) \
           + w[...,14] * (1 - (a * b)) \
           + w[...,15] * 1

    return output


def compute_weighted_logic_optim(a, b, weights):
    w = weights

    C_ab = (
        w[:,1]
        - w[:,2]
        - w[:,4]
        - 2*w[:,6]
        - w[:,7]
        + w[:,8]
        + 2*w[:,9]
        + w[:,11]
        + w[:,13]
        - w[:,14]
    )

    C_a = (
        w[:,2]
        + w[:,3]
        + w[:,6]
        + w[:,7]
        - w[:,8]
        - w[:,9]
        - w[:,12]
        - w[:,13]
    )

    C_b = (
        w[:,4]
        + w[:,5]
        + w[:,6]
        + w[:,7]
        - w[:,8]
        - w[:,9]
        - w[:,10]
        - w[:,11]
    )

    C_1 = (
        w[:,8]
        + w[:,9]
        + w[:,10]
        + w[:,11]
        + w[:,12]
        + w[:,13]
        + w[:,14]
        + w[:,15]
    )

    return (
        C_1.view(1,-1,1)
        + C_a.view(1,-1,1) * a
        + C_b.view(1,-1,1) * b
        + C_ab.view(1,-1,1) * (a * b)
    )


class LogicConvExplicitIndices(nn.Module):
    """
    Logic convolution layer using explicit index construction.
    All kernel positions are precomputed during initialization.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (batch, channels, height, width)
        receptive_field_shape: Tuple[int, int],
        indices_within_field: torch.Tensor,
        weights: torch.Tensor
    ):
        """
        Args:
            input_shape: (batch_size, in_channels, height, width) expected input shape
            receptive_field_shape: (height, width) of the receptive field
            indices_within_field: (2, n_kernels, 3) tensor with [h, w, c] indices
            weights: (n_kernels, 16) tensor of weights for logic operations
        """
        super().__init__()
        
        assert indices_within_field.dim() == 3, "Indices within field must dim 3"
        assert weights.dim() == 2, "Weights must have dim 2 (n_kernels, 16)"
        
        _, _, h, w = input_shape
        self.rcf_h, self.rcf_w = receptive_field_shape
        self.pairs_a = indices_within_field[0]  # (n_kernels, 3)
        self.pairs_b = indices_within_field[1]  # (n_kernels, 3)
        self.n_kernels = self.pairs_a.shape[0]
        
        assert weights.shape[0] == self.n_kernels, "Weights first dimension must match number of kernels"
        
        self.out_height = h - self.rcf_h + 1
        self.out_width = w - self.rcf_w + 1
        
        # Precompute all indices
        self._precompute_indices()
        
        # Register weights as buffer
        self.register_buffer('weights', weights)
        
    def _precompute_indices(self):
        """Precompute all kernel position indices."""
        h_starts = torch.arange(0, self.out_height)
        w_starts = torch.arange(0, self.out_width)
        
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")
        
        indices_a, indices_b = [], []
        for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
            idx_a = torch.stack([
                self.pairs_a[:, 0] + h_start,
                self.pairs_a[:, 1] + w_start,
                self.pairs_a[:, 2]
            ], dim=-1)
            idx_b = torch.stack([
                self.pairs_b[:, 0] + h_start,
                self.pairs_b[:, 1] + w_start,
                self.pairs_b[:, 2]
            ], dim=-1)
            
            indices_a.append(idx_a)
            indices_b.append(idx_b)
        
        indices_a = torch.stack(indices_a, dim=0)  # (num_positions, n_kernels, 3)
        indices_b = torch.stack(indices_b, dim=0)  # (num_positions, n_kernels, 3)
        
        # Extract and store separate h, w, c indices
        self.register_buffer('a_h', indices_a[..., 0])  # (num_positions, n_kernels)
        self.register_buffer('a_w', indices_a[..., 1])  # (num_positions, n_kernels)
        self.register_buffer('a_c', indices_a[..., 2])  # (num_positions, n_kernels)
        self.register_buffer('b_h', indices_b[..., 0])  # (num_positions, n_kernels)
        self.register_buffer('b_w', indices_b[..., 1])  # (num_positions, n_kernels)
        self.register_buffer('b_c', indices_b[..., 2])  # (num_positions, n_kernels)
        
    def forward(self, input: torch.Tensor, mode='ours') -> torch.Tensor:
        """
        Args:
            input: (batch_size, in_channels, height, width) tensor
            
        Returns:
            output: (batch_size, n_kernels, out_height, out_width) tensor
        """
        assert input.dim() == 4, "Input must be 4D tensor"
        
        batch_size = input.shape[0]
        
        a = input[:, self.a_c, self.a_h, self.a_w]  # (batch, num_positions, n_kernels)
        b = input[:, self.b_c, self.b_h, self.b_w]  # (batch, num_positions, n_kernels)
        
        a = a.transpose(1, 2)  # (batch, n_kernels, num_positions)
        b = b.transpose(1, 2)  # (batch, n_kernels, num_positions)

        if mode == 'ours':
            output = compute_all_logic_ops_vectorized(a, b)  # (batch, n_kernels, num_positions, 16)
            output = output * self.weights.view(1, self.n_kernels, 1, -1)
            output = output.sum(dim=-1)  # (batch, n_kernels, num_positions)
        elif mode == 'fused':
            output = compute_weighted_logic(a, b, self.weights)  # (batch, n_kernels, num_positions)
        elif mode == 'fused-optim':
            output = compute_weighted_logic_optim(a, b, self.weights)  # (batch, n_kernels, num_positions)

        # Reshape to (batch, n_kernels, out_height, out_width)
        output = output.view(batch_size, self.n_kernels, self.out_height, self.out_width)
        
        return output


class LogicConvUnfold(nn.Module):
    """
    Logic convolution layer using F.unfold for patch extraction.
    Output dimensions are precomputed during initialization.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (batch, channels, height, width)
        receptive_field_shape: Tuple[int, int],
        indices_within_field: torch.Tensor,
        weights: torch.Tensor
    ):
        """
        Args:
            input_shape: (batch_size, in_channels, height, width) expected input shape
            receptive_field_shape: (height, width) of the receptive field
            indices_within_field: (2, n_kernels, 3) tensor with [h, w, c] indices
            weights: (n_kernels, 16) tensor of weights for logic operations
        """
        super().__init__()
        
        assert indices_within_field.dim() == 3, "Indices within field must dim 3"
        assert weights.dim() == 2, "Weights must have dim 2 (n_kernels, 16)"
        
        _, in_channels, in_height, in_width = input_shape
        self.in_channels = in_channels
        self.rcf_h, self.rcf_w = receptive_field_shape
        self.pairs_a = indices_within_field[0]  # (n_kernels, 3)
        self.pairs_b = indices_within_field[1]  # (n_kernels, 3)
        self.n_kernels = self.pairs_a.shape[0]
        
        assert weights.shape[0] == self.n_kernels, "Weights first dimension must match number of kernels"
        
        self.out_height = in_height - self.rcf_h + 1
        self.out_width = in_width - self.rcf_w + 1
        self.num_positions = self.out_height * self.out_width
        
        self.register_buffer('weights', weights)
        
    def forward(self, input: torch.Tensor, mode='ours') -> torch.Tensor:
        """
        Args:
            input: (batch_size, in_channels, height, width) tensor
            
        Returns:
            output: (batch_size, n_kernels, out_height, out_width) tensor
        """
        assert input.dim() == 4, "Input must be 4D tensor"
        
        batch_size = input.shape[0]
        
        # Extract patches using unfold
        unfolded_input = F.unfold(input, (self.rcf_h, self.rcf_w))
        # Shape: (batch_size, in_channels * rcf_h * rcf_w, num_positions)
        
        # Reshape to (batch_size, in_channels, rcf_h, rcf_w, num_positions)
        unfolded_input = unfolded_input.view(
            batch_size, self.in_channels, self.rcf_h, self.rcf_w, self.num_positions
        )
        
        # Extract values at specified indices
        a = unfolded_input[:, self.pairs_a[:, 2], self.pairs_a[:, 0], self.pairs_a[:, 1], :]
        b = unfolded_input[:, self.pairs_b[:, 2], self.pairs_b[:, 0], self.pairs_b[:, 1], :]
        # Shape: (batch_size, n_kernels, num_positions)
        
        if mode == 'ours':
            output = compute_all_logic_ops_vectorized(a, b)  # (batch, n_kernels, num_positions, 16)
            output = output * self.weights.view(1, self.n_kernels, 1, -1)
            output = output.sum(dim=-1)  # (batch, n_kernels, num_positions)

        elif mode == 'fused':
            output = compute_weighted_logic(a, b, self.weights)  # (batch, n_kernels, num_positions)

        elif mode == 'fused-optim':
            output = compute_weighted_logic_optim(a, b, self.weights)  # (batch, n_kernels, num_positions)
        
        # Reshape to (batch, n_kernels, out_height, out_width)
        output = output.view(batch_size, self.n_kernels, self.out_height, self.out_width)
        
        return output


class LogicConvSparseMatrix(nn.Module):
    """
    Logic convolution layer using sparse matrix multiplication.
    Sparse matrices are precomputed during initialization.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (batch, channels, height, width)
        receptive_field_shape: Tuple[int, int],
        indices_within_field: torch.Tensor,
        weights: torch.Tensor
    ):
        """
        Args:
            input_shape: (batch_size, in_channels, height, width) expected input shape
            receptive_field_shape: (height, width) of the receptive field
            indices_within_field: (2, n_kernels, 3) tensor with [h, w, c] indices
            weights: (n_kernels, 16) tensor of weights for logic operations
        """
        super().__init__()
        
        assert indices_within_field.dim() == 3, "Indices within field must dim 3"
        assert weights.dim() == 2, "Weights must have dim 2 (n_kernels, 16)"
        
        _, in_channels, in_height, in_width = input_shape
        self.in_channels = in_channels
        self.rcf_h, self.rcf_w = receptive_field_shape
        self.pairs_a = indices_within_field[0]  # (n_kernels, 3)
        self.pairs_b = indices_within_field[1]  # (n_kernels, 3)
        self.n_kernels = self.pairs_a.shape[0]
        self.rf_volume = in_channels * self.rcf_h * self.rcf_w
        
        assert weights.shape[0] == self.n_kernels, "Weights first dimension must match number of kernels"
        
        self.out_height = in_height - self.rcf_h + 1
        self.out_width = in_width - self.rcf_w + 1
        self.num_positions = self.out_height * self.out_width
        
        # Precompute sparse matrices
        self._build_sparse_matrices()
        
        self.register_buffer('weights', weights)
        
    def _build_sparse_selector_multi_kernel(self, pair_indices: torch.Tensor):
        """
        Build a sparse matrix that selects elements from unfolded_input for multiple kernels.
        
        Args:
            pair_indices: (n_kernels, 3) tensor with [h, w, c] indices within receptive field
            
        Returns:
            sparse matrix of shape (n_kernels * num_positions, rf_volume * num_positions)
        """
        indices = []
        values = []
        
        # For each kernel
        for k in range(self.n_kernels):
            h_idx = pair_indices[k, 0].item()
            w_idx = pair_indices[k, 1].item()
            c_idx = pair_indices[k, 2].item()
            
            # Flat index in the unfolded representation (channel-major order)
            # unfold uses: c * (rcf_h * rcf_w) + h * rcf_w + w
            rf_flat_idx = c_idx * (self.rcf_h * self.rcf_w) + h_idx * self.rcf_w + w_idx
            
            # For each output position for this kernel
            for p in range(self.num_positions):
                row_idx = k * self.num_positions + p  # which (kernel, position) output
                col_idx = rf_flat_idx * self.num_positions + p  # index in flattened (rf_volume, num_positions)
                
                indices.append([row_idx, col_idx])
                values.append(1.0)
        
        indices_tensor = torch.tensor(indices).T
        values_tensor = torch.tensor(values, dtype=torch.float32)
        
        sparse_mat = torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            (self.n_kernels * self.num_positions, self.rf_volume * self.num_positions)
        )
        
        return sparse_mat.to_sparse_csr()
    
    def _build_sparse_matrices(self):
        """Precompute sparse selection matrices for both a and b indices."""
        sparse_a = self._build_sparse_selector_multi_kernel(self.pairs_a)
        sparse_b = self._build_sparse_selector_multi_kernel(self.pairs_b)
        
        # Register as buffers so they move with the module
        self.register_buffer('sparse_a', sparse_a)
        self.register_buffer('sparse_b', sparse_b)
    
    def forward(self, input: torch.Tensor, mode='ours') -> torch.Tensor:
        """
        Args:
            input: (batch_size, in_channels, height, width) tensor
            
        Returns:
            output: (batch_size, n_kernels, out_height, out_width) tensor
        """
        assert input.dim() == 4, "Input must be 4D tensor"
        
        batch_size, in_channels = input.shape[0], input.shape[1]
        assert in_channels == self.in_channels, f"Expected {self.in_channels} input channels, got {in_channels}"
        
        # Get patches using unfold
        unfolded_input = F.unfold(input, (self.rcf_h, self.rcf_w))
        # Shape: (batch, rf_volume, num_positions)
        
        # Flatten unfolded_input for sparse matmul
        unfolded_flat = unfolded_input.reshape(batch_size, -1)
        # Shape: (batch, rf_volume * num_positions)
        
        # Apply sparse matmul using precomputed sparse matrices
        a = torch.sparse.mm(self.sparse_a, unfolded_flat.T).T
        b = torch.sparse.mm(self.sparse_b, unfolded_flat.T).T
        # Shape: (batch, n_kernels * num_positions)
        
        # Reshape to (batch, n_kernels, num_positions)
        a = a.reshape(batch_size, self.n_kernels, self.num_positions)
        b = b.reshape(batch_size, self.n_kernels, self.num_positions)
        
        # Compute logic operations
        if mode == 'ours':
            output = compute_all_logic_ops_vectorized(a, b)  # (batch, n_kernels, num_positions, 16)
            output = output * self.weights.view(1, self.n_kernels, 1, -1)
            output = output.sum(dim=-1)  # (batch, n_kernels, num_positions)
        elif mode == 'fused':
            output = compute_weighted_logic(a, b, self.weights)  # (batch, n_kernels, num_positions)
        elif mode == 'fused-optim':
            output = compute_weighted_logic_optim(a, b, self.weights)  # (batch, n_kernels, num_positions)
        
        # Reshape to (batch, n_kernels, out_height, out_width)
        output = output.view(batch_size, self.n_kernels, self.out_height, self.out_width)
        
        return output


def print_memory_usage(stage_name):
    allocated = torch.cuda.memory_allocated(0) / 1e6
    reserved = torch.cuda.memory_reserved(0) / 1e6
    print(f"{stage_name:30s} | Allocated: {allocated:6.2f} MB | Reserved: {reserved:6.2f} MB")


def bench(
    layer,
    input_shape,
    device,
    mode='ours',
    n_runs=30,
    n_warmup=5,
    dtype=torch.float32,
):
    """
    Benchmarks a layer for:
      - average forward time (ms)
      - peak allocated memory (MB)
      - peak reserved memory (MB)
    
    Returns a dict with results.
    """

    assert device.type == "cuda", "This benchmark is intended for CUDA"

    # -----------------------------
    # Warm-up (no measurement)
    # -----------------------------
    for _ in range(n_warmup):
        x = torch.randn(*input_shape, device=device, dtype=dtype)
        _ = layer(x, mode=mode)
    torch.cuda.synchronize()

    # -----------------------------
    # Reset memory stats
    # -----------------------------
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # -----------------------------
    # Timed runs
    # -----------------------------
    start = time.perf_counter()
    for _ in range(n_runs):
        x = torch.randn(*input_shape, device=device, dtype=dtype)
        _ = layer(x, mode=mode)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_runs * 1000.0

    # -----------------------------
    # Memory stats
    # -----------------------------
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1e6
    peak_reserved_mb  = torch.cuda.max_memory_reserved(device) / 1e6

    print(f"{layer.__class__.__name__:25s} | Mode: {mode:10s} | Time: {avg_time_ms:6.2f} ms | "
          f"Peak Allocated: {peak_allocated_mb:6.2f} MB | "
          f"Peak Reserved: {peak_reserved_mb:6.2f} MB")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define input shape
    input_shape = (256, 3, 32, 32)  # Batch size of 256, 3 channels, 32x32 image

    # Generate an example batch of images like CIFAR-10
    input_tensor = torch.randn(*input_shape, device=device)
    
    rfs = (3, 3)
    indices_within_field = torch.randint(0, 3, (2, 100, 3))  # 100 kernels
    weights = torch.randn(indices_within_field.shape[1], 16)
    
    # Create layer instances with input shape
    logic_conv_explicit = LogicConvExplicitIndices(
        input_shape, rfs, indices_within_field, weights
    ).to(device)
    print_memory_usage("After logic conv explicit init")
    
    logic_conv_unfold = LogicConvUnfold(
        input_shape, rfs, indices_within_field, weights
    ).to(device)
    print_memory_usage("After logic conv unfold init")
    
    logic_conv_sparse = LogicConvSparseMatrix(
        input_shape, rfs, indices_within_field, weights
    ).to(device)
    print_memory_usage("After logic conv sparse init")

    # Test functional equivalence    
    output_explicit = logic_conv_explicit(input_tensor)
    output_unfold = logic_conv_unfold(input_tensor)
    output_sparse = logic_conv_sparse(input_tensor)
    
    print(f"\nLogic conv explicit output shape: {output_explicit.shape}")
    print(f"Logic conv unfold output shape: {output_unfold.shape}")
    print(f"Logic conv sparse output shape: {output_sparse.shape}")
    
    print("\nComparison:")
    if torch.allclose(output_explicit, output_unfold, atol=1e-5):
        print("✓ Explicit and unfold outputs are the same!")
    else:
        print("✗ Explicit and unfold outputs are different!")
        
    if torch.allclose(output_explicit, output_sparse, atol=1e-5):
        print("✓ Explicit and sparse outputs are the same!")
    else:
        print("✗ Explicit and sparse outputs are different!")

    output_ours = logic_conv_explicit(input_tensor, mode='ours')
    output_fused = logic_conv_explicit(input_tensor, mode='fused')
    output_fused_optim = logic_conv_explicit(input_tensor, mode='fused-optim')

    print("\nMode comparison (Explicit):")
    if torch.allclose(output_ours, output_fused, atol=1e-5):
        print("✓ 'ours' and 'fused' outputs are the same!")
    else:
        print("✗ 'ours' and 'fused' outputs are different!")
    if torch.allclose(output_ours, output_fused_optim, atol=1e-5):
        print("✓ 'ours' and 'fused-optim' outputs are the same!")
    else:
        print("✗ 'ours' and 'fused-optim' outputs are different!")


    # memory and time benchmarks
    for mode in ['ours', 'fused', 'fused-optim']:
        bench(logic_conv_explicit, input_shape, device=device, mode=mode)
        bench(logic_conv_unfold, input_shape, device=device, mode=mode)
        bench(logic_conv_sparse, input_shape, device=device, mode=mode)

