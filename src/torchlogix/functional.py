"""Functional operations for differentiable logic neural networks.

This module provides the core mathematical operations for computing logic neural
operations in a differentiable manner. It includes implementations for binary
operations, vectorized operations, and utility functions for building logic
gate networks.
"""
import math
import random
import torch
import numpy as np


BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# The 16 possible binary logic operations on two inputs A and B
# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 | Walsh Coefficients      |
# |----|----------------------|-------|-------|-------|-------|-------------------------|
# | 0  | 0                    | 0     | 0     | 0     | 0     | (1, 0, 0, 0)            |
# | 1  | A and B              | 0     | 0     | 0     | 1     | (0.5, 0.5, 0.5, -0.5)   |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     | (0.5, -0.5, 0.5, 0.5)   |
# | 3  | A                    | 0     | 0     | 1     | 1     | (0, 0, 1, 0)            |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     | (0.5, 0.5, -0.5, 0.5)   |
# | 5  | B                    | 0     | 1     | 0     | 1     | (0, 1, 0, 0)            |
# | 6  | A xor B              | 0     | 1     | 1     | 0     | (0, 0, 0, 1)            |
# | 7  | A or B               | 0     | 1     | 1     | 1     | (-0.5, 0.5, 0.5, 0.5)   |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     | (0.5, -0.5, -0.5, -0.5) |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     | (0, 0, 0, -1)           |
# | 10 | not(B)               | 1     | 0     | 1     | 0     | (0, -1, 0, 0)           |
# | 11 | B implies A          | 1     | 0     | 1     | 1     | (-0.5, -0.5, 0.5, -0.5) |
# | 12 | not(A)               | 1     | 1     | 0     | 0     | (0, 0, -1, 0)           |
# | 13 | A implies B          | 1     | 1     | 0     | 1     | (-0.5, 0.5, -0.5, -0.5) |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     | (-0.5, -0.5, -0.5, 0.5) |
# | 15 | 1                    | 1     | 1     | 1     | 1     | (-1, 0, 0, 0)           |


ID_TO_OP = {
    0: lambda a, b: torch.zeros_like(a),
    1: lambda a, b: a * b,
    2: lambda a, b: a - a * b,
    3: lambda a, b: a,
    4: lambda a, b: b - a * b,
    5: lambda a, b: b,
    6: lambda a, b: a + b - 2 * a * b,
    7: lambda a, b: a + b - a * b,
    8: lambda a, b: 1 - (a + b - a * b),
    9: lambda a, b: 1 - (a + b - 2 * a * b),
    10: lambda a, b: 1 - b,
    11: lambda a, b: 1 - b + a * b,
    12: lambda a, b: 1 - a,
    13: lambda a, b: 1 - a + a * b,
    14: lambda a, b: 1 - a * b,
    15: lambda a, b: torch.ones_like(a),
}

ID_TO_WALSH_COEFFICIENTS = {
    0: (1, 0, 0, 0),
    1: (0.5, 0.5, 0.5, -0.5),
    2: (0.5, -0.5, 0.5, 0.5),
    3: (0, 0, 1, 0),
    4: (0.5, 0.5, -0.5, 0.5),
    5: (0, 1, 0, 0),
    6: (0, 0, 0, 1),
    7: (-0.5, 0.5, 0.5, 0.5),
    8: (0.5, -0.5, -0.5, -0.5),
    9: (0, 0, 0, -1),
    10: (0, -1, 0, 0),
    11: (-0.5, -0.5, 0.5, -0.5),
    12: (0, 0, -1, 0),
    13: (-0.5, 0.5, -0.5, -0.5),
    14: (-0.5, -0.5, -0.5, 0.5),
    15: (-1, 0, 0, 0),
}


def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)
    return ID_TO_OP[i](a, b)


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
        torch.zeros_like(a),           # 0: 0
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


def apply_luts_vectorized_export_mode(
    a: torch.BoolTensor | np.ndarray[bool],
    b: torch.BoolTensor | np.ndarray[bool],
    lut_ids: torch.IntTensor | np.ndarray[int]
) -> torch.Tensor | np.ndarray:
    """Tracer-friendly LUT application for ONNX/TorchScript export.

    Works with both torch tensors and numpy arrays. Detects backend once at entry
    and routes to the appropriate implementation.

    Args:
        a: First input (batch, neurons) or (batch, spatial, kernels, features)
        b: Second input (same shape as a)
        lut_ids: LUT operation IDs (neurons,) or (kernels, features)

    Returns:
        Result with same shape and backend as inputs

    Note: This function is optimized for export/tracing, not runtime performance.
          For training/eval without export, use the regular forward pass.
    """
    # Smart dispatch - single isinstance check
    if isinstance(a, np.ndarray):
        return _apply_luts_export_numpy(a, b, lut_ids)
    else:
        return _apply_luts_export_torch(a, b, lut_ids)


def _apply_luts_export_torch(
    a: torch.BoolTensor,
    b: torch.BoolTensor,
    lut_ids: torch.IntTensor
) -> torch.Tensor:
    """Torch implementation using torch.where for ONNX/TorchScript tracing.

    Args:
        a: First input tensor (batch, neurons) or (batch, spatial, kernels, features)
        b: Second input tensor (same shape as a)
        lut_ids: LUT operation IDs (neurons,) or (kernels, features)

    Returns:
        Float32 result tensor (0.0 or 1.0)
    """
    # ensure inputs are boolean
    a = a.bool()
    b = b.bool()
    
    # Broadcast lut_ids to match input shape for element-wise comparison
    while lut_ids.ndim < a.ndim:
        lut_ids = lut_ids.unsqueeze(0)

    # Initialize result with all False
    result = torch.zeros_like(a, dtype=torch.bool)

    # Unroll all 16 LUT operations using torch.where
    # This is fully traceable by ONNX/TorchScript
    result = torch.where(lut_ids == 0, torch.zeros_like(a, dtype=torch.bool), result)           # FALSE
    result = torch.where(lut_ids == 1, a & b, result)               # AND
    result = torch.where(lut_ids == 2, a & ~b, result)              # A AND NOT B
    result = torch.where(lut_ids == 3, a, result)                   # A
    result = torch.where(lut_ids == 4, b & ~a, result)              # B AND NOT A
    result = torch.where(lut_ids == 5, b, result)                   # B
    result = torch.where(lut_ids == 6, a ^ b, result)               # XOR
    result = torch.where(lut_ids == 7, a | b, result)               # OR
    result = torch.where(lut_ids == 8, ~(a | b), result)            # NOR
    result = torch.where(lut_ids == 9, ~(a ^ b), result)            # XNOR
    result = torch.where(lut_ids == 10, ~b, result)                 # NOT B
    result = torch.where(lut_ids == 11, ~b | a, result)             # B IMPLIES A
    result = torch.where(lut_ids == 12, ~a, result)                 # NOT A
    result = torch.where(lut_ids == 13, ~a | b, result)             # A IMPLIES B
    result = torch.where(lut_ids == 14, ~(a & b), result)           # NAND
    result = torch.where(lut_ids == 15, torch.ones_like(a, dtype=torch.bool), result)           # TRUE

    return result


def _apply_luts_export_numpy(
    a: np.ndarray[bool],
    b: np.ndarray[bool],
    lut_ids: np.ndarray[int]
) -> np.ndarray:
    """NumPy implementation using boolean operations for 3rd party tracers.

    Args:
        a: First input array (batch, neurons) or (batch, spatial, kernels, features)
        b: Second input array (same shape as a)
        lut_ids: LUT operation IDs (neurons,) or (kernels, features)

    Returns:
        Float32 result array (0.0 or 1.0)
    """
    # Broadcast lut_ids to match input shape for element-wise comparison
    while lut_ids.ndim < a.ndim:
        lut_ids = np.expand_dims(lut_ids, axis=0)

    # Initialize result with all False
    result = np.zeros_like(a, dtype=bool)

    # Unroll all 16 LUT operations using np.where
    result = np.where(lut_ids == 0, np.zeros_like(a, dtype=bool), result)  # FALSE
    result = np.where(lut_ids == 1, a & b, result)                          # AND
    result = np.where(lut_ids == 2, a & ~b, result)                         # A AND NOT B
    result = np.where(lut_ids == 3, a, result)                              # A
    result = np.where(lut_ids == 4, b & ~a, result)                         # B AND NOT A
    result = np.where(lut_ids == 5, b, result)                              # B
    result = np.where(lut_ids == 6, a ^ b, result)                          # XOR
    result = np.where(lut_ids == 7, a | b, result)                          # OR
    result = np.where(lut_ids == 8, ~(a | b), result)                       # NOR
    result = np.where(lut_ids == 9, ~(a ^ b), result)                       # XNOR
    result = np.where(lut_ids == 10, ~b, result)                            # NOT B
    result = np.where(lut_ids == 11, ~b | a, result)                        # B IMPLIES A
    result = np.where(lut_ids == 12, ~a, result)                            # NOT A
    result = np.where(lut_ids == 13, ~a | b, result)                        # A IMPLIES B
    result = np.where(lut_ids == 14, ~(a & b), result)                      # NAND
    result = np.where(lut_ids == 15, np.ones_like(a, dtype=bool), result)   # TRUE

    return result


def weighted_raw_basis_sum(a, b, weights, einsum_pattern) -> torch.Tensor:
    """
    Compute the weighted sum of the raw basis functions.
    The 16-term weighted sum can be expressed using only
    4 coefficients, which is computationally more efficient.
    """
    w = weights

    C1 = (
        w[...,8] + w[...,9] + w[...,10] + w[...,11] +
        w[...,12] + w[...,13] + w[...,14] + w[...,15]
    )

    Ca = (
        w[...,2] + w[...,3] + w[...,6] + w[...,7]
        - w[...,8] - w[...,9] - w[...,12] - w[...,13]
    )

    Cb = (
        w[...,4] + w[...,5] + w[...,6] + w[...,7]
        - w[...,8] - w[...,9] - w[...,10] - w[...,11]
    )

    Cab = (
        w[...,1] - w[...,2] - w[...,4]
        - 2*w[...,6] - w[...,7]
        + w[...,8] + 2*w[...,9]
        + w[...,11] + w[...,13] - w[...,14]
    )

    return (
        torch.einsum(einsum_pattern, C1, a * 0 + 1) +
        torch.einsum(einsum_pattern, Ca, a) +
        torch.einsum(einsum_pattern, Cb, b) +
        torch.einsum(einsum_pattern, Cab, a * b)
    )


##########################################################################


class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None


##########################################################################

def gumbel_sigmoid(logits, tau=1.0, hard=False, threshold=0.5):
    """
    Fast Gumbel-Sigmoid implementation using logistic noise trick.
    """
    if tau <= 0:
        raise ValueError("Temperature must be positive")

    # Logistic(0,1) noise from uniform: log(U) - log(1-U)
    U = torch.rand_like(logits)
    logistic_noise = torch.log(U + 1e-20) - torch.log(1 - U + 1e-20)

    # Soft sample
    y_soft = torch.sigmoid((logits + logistic_noise) / tau)

    if hard:
        # Straight-through estimator
        y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft

    return y_soft

def softmax(logits, hard=False, tau=1.0, dim=-1):
    y_soft = torch.nn.functional.softmax(logits / tau, dim=dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft

def sigmoid(logits, hard=False, tau=1.0):
    y_soft = torch.sigmoid(logits / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
    return y_soft

##################################################################

def fwht(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Fast Walsh–Hadamard transform on the last dimension.
    x: (..., n) with n a power of 2
    """
    lut_entries = x.size(-1)
    y = x.reshape(-1, lut_entries)  # collapse batch dims
    for s in range(n):
        step = 1 << s       # 2^s
        block = step << 1   # 2^(s+1)

        # shape: (batch, n / block, block)
        y = y.view(-1, lut_entries // block, block)

        a = y[..., :step]
        b = y[..., step:block]

        y = torch.cat((a + b, a - b), dim=-1)  # still (batch, n/block, block)
        y = y.reshape(-1, lut_entries)

    return y.view_as(x)


def hadamard_matrix(n, dtype=torch.float32, device=None):
    if n == 1:
        return torch.tensor([[1.0]], dtype=dtype, device=device)

    H = hadamard_matrix(n // 2, dtype=dtype, device=device)
    top = torch.cat([H, H], dim=1)
    bottom = torch.cat([H, -H], dim=1)
    return torch.cat([top, bottom], dim=0)


def walsh_hadamard_transform(x: torch.Tensor, n: int, dtype=torch.float32, 
                             device=None, fast=False) -> torch.Tensor:
    """
    Walsh-Hadamard transform on the last dimension.
    x: (..., 2^n)
    """
    if fast:
        return fwht(x, n)
    else:
       # Always use float32 for matrix multiplication
        # (CUDA doesn't support int32 matmul)
        # We'll convert the result back to the requested dtype if needed
        # Use device from input tensor if device parameter is None
        if device is None:
            device = x.device
        H = hadamard_matrix(1 << n, dtype=torch.float32, device=device)
        x_float = x.to(dtype=torch.float32)
        result = x_float @ H
        # Convert result back to requested dtype if different from float32
        if dtype != torch.float32:
            result = result.to(dtype=dtype)
        return result
        
##########################################################################
    
# Needs to be CUDAized
def kron_pairwise_basis(x):
    """
    # x: (..., n) where x[...,k] is the kth scalar input
    # returns: (..., 2**n) basis vector 
    #          matching EXACTLY the example:
    #          [1, A] x [1, B] = [1, B, A, A*B]
    #          etc.
    """
    *batch, n = x.shape
    m = 1 << n
    out = x.new_empty(*batch, m)
    out[..., 0] = 1

    size = 1
    for k in range(n):
        prev = out[..., :size]

        # Create next
        # Correct Kronecker ordering:
        # [prev*1, prev*x_k], but interleaved rather than blocked
        out[..., : 2*size : 2] = prev
        out[..., 1 : 2*size : 2] = prev * x[..., k].unsqueeze(-1)

        size *= 2

    return out


def walsh_basis_hard(x, lut_rank):
    if lut_rank == 1:
        A = x[:, 0]
        basis = walsh_basis_1(A)
    elif lut_rank == 2:
        A, B = x[:, 0], x[:, 1]
        basis = walsh_basis_2(A, B)
    elif lut_rank == 4:
        A, B, C, D = (x[:, 0], x[:, 1],
                        x[:, 2], x[:, 3]
                        )
        basis = walsh_basis_4(A, B, C, D)
    elif lut_rank == 6:
        A, B, C, D, E, F = (
            x[:, 0], x[:, 1], x[:, 2],
            x[:, 3], x[:, 4], x[:, 5],
        )
        basis = walsh_basis_6(A, B, C, D, E, F)
    else:
        raise ValueError(f"Hard basis not supported for lut_rank={lut_rank}")
    return basis


def weighted_walsh_basis_sum(x, weights, einsum_pattern, lut_rank) -> torch.Tensor:
    if lut_rank == 1:
        A = x[:, 0]
        basis = weighted_walsh_basis_sum_1(A, weights, einsum_pattern)
    elif lut_rank == 2:
        A, B = x[:, 0], x[:, 1]
        basis = weighted_walsh_basis_sum_2(A, B, weights, einsum_pattern)
    elif lut_rank == 4:
        A, B, C, D = (x[:, 0], x[:, 1],
                        x[:, 2], x[:, 3]
                        )
        basis = weighted_walsh_basis_sum_4(A, B, C, D, weights, einsum_pattern)
    elif lut_rank == 6:
        A, B, C, D, E, F = (
            x[:, 0], x[:, 1], x[:, 2],
            x[:, 3], x[:, 4], x[:, 5],
        )
        basis = weighted_walsh_basis_sum_6(A, B, C, D, E, F, weights, einsum_pattern)
    else:
        raise ValueError(f"Hard basis not supported for lut_rank={lut_rank}")
    return basis


def walsh_basis_1(A) -> torch.Tensor:
    #A = 1- 2 * a
    basis = torch.stack([
        torch.ones_like(A),
        A,
    ], dim=-1)
    return basis


def weighted_walsh_basis_sum_1(A, weights, einsum_pattern) -> torch.Tensor:
    return (
        torch.einsum(einsum_pattern, weights[...,0], A * 0 + 1) +
        torch.einsum(einsum_pattern, weights[...,1], A)
    )


def walsh_basis_2(A, B) -> torch.Tensor:
    #A = 1- 2 * a
    #B = 1 - 2 * b
    basis = torch.stack([
        torch.ones_like(A),
        B,
        A,
        A*B
    ], dim=-1)
    return basis


def weighted_walsh_basis_sum_2(A, B, weights, einsum_pattern) -> torch.Tensor:
    return (
        torch.einsum(einsum_pattern, weights[...,0], A * 0 + 1) +
        torch.einsum(einsum_pattern, weights[...,1], B) +
        torch.einsum(einsum_pattern, weights[...,2], A) +
        torch.einsum(einsum_pattern, weights[...,3], A * B)
    )


def walsh_basis_3(A, B, C) -> torch.Tensor:
    basis = torch.stack([
        torch.ones_like(A),
        C,
        B,
        B*C,
        A,
        A*C,
        A*B,
        A*B*C
    ], dim=-1)
    return basis


def walsh_basis_4(A, B, C, D) -> torch.Tensor:
    basis = torch.stack([
        torch.ones_like(A),
        D,
        C,
        C*D,
        B,
        B*D,
        B*C,
        B*C*D,
        A,
        A*D,
        A*C,
        A*C*D,
        A*B,
        A*B*D,
        A*B*C,
        A*B*C*D
    ], dim=-1)
    return basis


def weighted_walsh_basis_sum_4(A, B, C, D, weights, einsum_pattern) -> torch.Tensor:
    AB = A * B
    CD = C * D
    return (
        torch.einsum(einsum_pattern, weights[...,0], A * 0 + 1) +
        torch.einsum(einsum_pattern, weights[...,1], D) +
        torch.einsum(einsum_pattern, weights[...,2], C) +
        torch.einsum(einsum_pattern, weights[...,3], CD) +
        torch.einsum(einsum_pattern, weights[...,4], B) +
        torch.einsum(einsum_pattern, weights[...,5], B * D) +
        torch.einsum(einsum_pattern, weights[...,6], B * C) +
        torch.einsum(einsum_pattern, weights[...,7], B * CD) +
        torch.einsum(einsum_pattern, weights[...,8], A) +
        torch.einsum(einsum_pattern, weights[...,9], A * D) +
        torch.einsum(einsum_pattern, weights[...,10], A * C) +
        torch.einsum(einsum_pattern, weights[...,11], A * CD) +
        torch.einsum(einsum_pattern, weights[...,12], AB) +
        torch.einsum(einsum_pattern, weights[...,13], AB * D) +
        torch.einsum(einsum_pattern, weights[...,14], AB * C) +
        torch.einsum(einsum_pattern, weights[...,15], AB * CD)
    )


def walsh_basis_6(A, B, C, D, E, F) -> torch.Tensor:
    basis = torch.stack([
        torch.ones_like(A),
        F,
        E,
        E*F,
        D,
        D*F,
        D*E,
        D*E*F,
        C,
        C*F,
        C*E,
        C*E*F,
        C*D,
        C*D*F,
        C*D*E,
        C*D*E*F,
        B,
        B*F,
        B*E,
        B*E*F,
        B*D,
        B*D*F,
        B*D*E,
        B*D*E*F,
        B*C,
        B*C*F,
        B*C*E,
        B*C*E*F,
        B*C*D,
        B*C*D*F,
        B*C*D*E,
        B*C*D*E*F,
        A,
        A*F,
        A*E,
        A*E*F,
        A*D,
        A*D*F,
        A*D*E,
        A*D*E*F,
        A*C,
        A*C*F,
        A*C*E,
        A*C*E*F,
        A*C*D,
        A*C*D*F,
        A*C*D*E,
        A*C*D*E*F,
        A*B,
        A*B*F,
        A*B*E,
        A*B*E*F,
        A*B*D,
        A*B*D*F,
        A*B*D*E,
        A*B*D*E*F,
        A*B*C,
        A*B*C*F,
        A*B*C*E,
        A*B*C*E*F,
        A*B*C*D,
        A*B*C*D*F,
        A*B*C*D*E,
        A*B*C*D*E*F
    ], dim=-1)
    return basis


def weighted_walsh_basis_sum_6(A, B, C, D, E, F, weights, einsum_pattern) -> torch.Tensor:
    AB = A * B
    CD = C * D
    EF = E * F
    AC = A * C
    AD = A * D
    AE = A * E
    return (
        torch.einsum(einsum_pattern, weights[...,0], A * 0 + 1) +
        torch.einsum(einsum_pattern, weights[...,1], F) +
        torch.einsum(einsum_pattern, weights[...,2], E) +
        torch.einsum(einsum_pattern, weights[...,3], EF) +
        torch.einsum(einsum_pattern, weights[...,4], D) +
        torch.einsum(einsum_pattern, weights[...,5], D * F) +
        torch.einsum(einsum_pattern, weights[...,6], D * E) +
        torch.einsum(einsum_pattern, weights[...,7], D * EF) +
        torch.einsum(einsum_pattern, weights[...,8], C) +
        torch.einsum(einsum_pattern, weights[...,9], C * F) +
        torch.einsum(einsum_pattern, weights[...,10], C * E) +
        torch.einsum(einsum_pattern, weights[...,11], C * EF) +
        torch.einsum(einsum_pattern, weights[...,12], CD) +
        torch.einsum(einsum_pattern, weights[...,13], CD * F) +
        torch.einsum(einsum_pattern, weights[...,14], CD * E) +
        torch.einsum(einsum_pattern, weights[...,15], CD * EF) +
        torch.einsum(einsum_pattern, weights[...,16], B) +
        torch.einsum(einsum_pattern, weights[...,17], B * F) +
        torch.einsum(einsum_pattern, weights[...,18], B * E) +
        torch.einsum(einsum_pattern, weights[...,19], B * E * F) +
        torch.einsum(einsum_pattern, weights[...,20], B * D) +
        torch.einsum(einsum_pattern, weights[...,21], B * D * F) +
        torch.einsum(einsum_pattern, weights[...,22], B * D * E) +
        torch.einsum(einsum_pattern, weights[...,23], B * D * EF) +
        torch.einsum(einsum_pattern, weights[...,24], B * C) +
        torch.einsum(einsum_pattern, weights[...,25], B * C * F) +
        torch.einsum(einsum_pattern, weights[...,26], B * C * E) +
        torch.einsum(einsum_pattern, weights[...,27], B * C * EF) +
        torch.einsum(einsum_pattern, weights[...,28], B * CD) +
        torch.einsum(einsum_pattern, weights[...,29], B * CD * F) +
        torch.einsum(einsum_pattern, weights[...,30], B * CD * E) +
        torch.einsum(einsum_pattern, weights[...,31], B * CD * EF) +
        torch.einsum(einsum_pattern, weights[...,32], A) +
        torch.einsum(einsum_pattern, weights[...,33], A * F) +
        torch.einsum(einsum_pattern, weights[...,34], AE) +
        torch.einsum(einsum_pattern, weights[...,35], AE * F) +
        torch.einsum(einsum_pattern, weights[...,36], AD) +
        torch.einsum(einsum_pattern, weights[...,37], AD * F) +
        torch.einsum(einsum_pattern, weights[...,38], AD * E) +
        torch.einsum(einsum_pattern, weights[...,39], AD * EF) +
        torch.einsum(einsum_pattern, weights[...,40], AC) +
        torch.einsum(einsum_pattern, weights[...,41], AC * F) +
        torch.einsum(einsum_pattern, weights[...,42], AC * E) +
        torch.einsum(einsum_pattern, weights[...,43], AC * EF) +
        torch.einsum(einsum_pattern, weights[...,44], AC * D) +
        torch.einsum(einsum_pattern, weights[...,45], A * CD * F) +
        torch.einsum(einsum_pattern, weights[...,46], AE * CD) +
        torch.einsum(einsum_pattern, weights[...,47], A * CD * EF) +
        torch.einsum(einsum_pattern, weights[...,48], AB) +
        torch.einsum(einsum_pattern, weights[...,49], AB * F) +
        torch.einsum(einsum_pattern, weights[...,50], AB * E) +
        torch.einsum(einsum_pattern, weights[...,51], AB * E * F) +
        torch.einsum(einsum_pattern, weights[...,52], AB * D) +
        torch.einsum(einsum_pattern, weights[...,53], AB * D * F) +
        torch.einsum(einsum_pattern, weights[...,54], AB * D * E) +
        torch.einsum(einsum_pattern, weights[...,55], AB * D * EF) +
        torch.einsum(einsum_pattern, weights[...,56], AB * C) +
        torch.einsum(einsum_pattern, weights[...,57], AB * C * F) +
        torch.einsum(einsum_pattern, weights[...,58], AB * C * E) +
        torch.einsum(einsum_pattern, weights[...,59], AB * C * EF) +
        torch.einsum(einsum_pattern, weights[...,60], AB * CD) +
        torch.einsum(einsum_pattern, weights[...,61], AB * CD * F) +
        torch.einsum(einsum_pattern, weights[...,62], AB * CD * E) +
        torch.einsum(einsum_pattern, weights[...,63], AB * CD * EF)
    )


def light_basis_hard(x, lut_rank):
    if lut_rank == 2:
        A, B = x[:, 0], x[:, 1]
        basis = light_basis_2(A, B)
    elif lut_rank == 4:
        A, B, C, D = (x[:, 0], x[:, 1],
                        x[:, 2], x[:, 3]
                        )
        basis = light_basis_4(A, B, C, D)
    elif lut_rank == 6:
        A, B, C, D, E, F = (
            x[:, 0], x[:, 1], x[:, 2],
            x[:, 3], x[:, 4], x[:, 5],
        )
        basis = light_basis_6(A, B, C, D, E, F)
    else:
        raise ValueError(f"Hard basis not supported for lut_rank={lut_rank}")
    return basis


def weighted_light_basis_sum(x, weights, einsum_pattern, lut_rank) -> torch.Tensor:
    if lut_rank == 2:
        A, B = x[:, 0], x[:, 1]
        basis = weighted_light_basis_sum_2(A, B, weights, einsum_pattern)
    elif lut_rank == 4:
        A, B, C, D = (x[:, 0], x[:, 1],
                        x[:, 2], x[:, 3]
                        )
        basis = weighted_light_basis_sum_4(A, B, C, D, weights, einsum_pattern)
    elif lut_rank == 6:
        A, B, C, D, E, F = (
            x[:, 0], x[:, 1], x[:, 2],
            x[:, 3], x[:, 4], x[:, 5],
        )
        basis = weighted_light_basis_sum_6(A, B, C, D, E, F, weights, einsum_pattern)
    else:
        raise ValueError(f"Hard basis not supported for lut_rank={lut_rank}")
    return basis


def light_basis_2(A, B) -> torch.Tensor:
    basis = torch.stack([
        (1 - A) * (1 - B),
        (1 - A) * B,
        A * (1 - B),
        A*B
    ], dim=-1)
    return basis


def weighted_light_basis_sum_2(A, B, weights, einsum_pattern) -> torch.Tensor:
    return (
        torch.einsum(einsum_pattern, weights[...,0], (1 - A) * (1 - B)) +
        torch.einsum(einsum_pattern, weights[...,1], (1 - A) * B) +
        torch.einsum(einsum_pattern, weights[...,2], A * (1 - B)) +
        torch.einsum(einsum_pattern, weights[...,3], A * B)
    )

def light_basis_3(A, B, C) -> torch.Tensor:
    basis = torch.stack([
        (1 - A) * (1 - B) * (1 - C),
        (1 - A) * (1 - B) * C,
        (1 - A) * B * (1 - C),
        (1 - A) * B * C,
        A * (1 - B) * (1 - C),
        A * (1 - B) * C,
        A*B * (1 - C),
        A*B*C
    ], dim=-1)
    return basis

def light_basis_4(A, B, C, D) -> torch.Tensor:
    basis = torch.stack([
        (1 - A) * (1 - B) * (1 - C) * (1 - D),
        (1 - A) * (1 - B) * (1 - C) * D,
        (1 - A) * (1 - B) * C * (1 - D),
        (1 - A) * (1 - B) * C * D,
        (1 - A) * B * (1 - C) * (1 - D),
        (1 - A) * B * (1 - C) * D,
        (1 - A) * B * C * (1 - D),
        (1 - A) * B * C * D,
        A * (1 - B) * (1 - C) * (1 - D),
        A * (1 - B) * (1 - C) * D,
        A * (1 - B) * C * (1 - D),
        A * (1 - B) * C * D,
        A * B * (1 - C) * (1 - D),
        A * B * (1 - C) * D,
        A * B * C * (1 - D),
        A * B * C * D
    ], dim=-1)
    return basis


def weighted_light_basis_sum_4(A, B, C, D, weights, einsum_pattern) -> torch.Tensor:
    return (
        torch.einsum(einsum_pattern, weights[...,0], (1 - A) * (1 - B) * (1 - C) * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,1], (1 - A) * (1 - B) * (1 - C) * D) +
        torch.einsum(einsum_pattern, weights[...,2], (1 - A) * (1 - B) * C * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,3], (1 - A) * (1 - B) * C * D) +
        torch.einsum(einsum_pattern, weights[...,4], (1 - A) * B * (1 - C) * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,5], (1 - A) * B * (1 - C) * D) +
        torch.einsum(einsum_pattern, weights[...,6], (1 - A) * B * C * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,7], (1 - A) * B * C * D) +
        torch.einsum(einsum_pattern, weights[...,8], A * (1 - B) * (1 - C) * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,9], A * (1 - B) * (1 - C) * D) +
        torch.einsum(einsum_pattern, weights[...,10], A * (1 - B) * C * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,11], A * (1 - B) * C * D) +
        torch.einsum(einsum_pattern, weights[...,12], A * B * (1 - C) * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,13], A * B * (1 - C) * D) +
        torch.einsum(einsum_pattern, weights[...,14], A * B * C * (1 - D)) +
        torch.einsum(einsum_pattern, weights[...,15], A * B * C * D)
    )


def light_basis_6(A, B, C, D, E, F) -> torch.Tensor:
    basis = torch.stack([
        (1 - A) * (1 - B) * (1 - C) * (1 - D) * (1 - E) * (1 - F),
        (1 - A) * (1 - B) * (1 - C) * (1 - D) * (1 - E) * F,
        (1 - A) * (1 - B) * (1 - C) * (1 - D) * E * (1 - F),
        (1 - A) * (1 - B) * (1 - C) * (1 - D) * E * F,
        (1 - A) * (1 - B) * (1 - C) * D * (1 - E) * (1 - F),
        (1 - A) * (1 - B) * (1 - C) * D * (1 - E) * F,
        (1 - A) * (1 - B) * (1 - C) * D * E * (1 - F),
        (1 - A) * (1 - B) * (1 - C) * D * E * F,
        (1 - A) * (1 - B) * C * (1 - D) * (1 - E) * (1 - F),
        (1 - A) * (1 - B) * C * (1 - D) * (1 - E) * F,
        (1 - A) * (1 - B) * C * (1 - D) * E * (1 - F),
        (1 - A) * (1 - B) * C * (1 - D) * E * F,
        (1 - A) * (1 - B) * C * D * (1 - E) * (1 - F),
        (1 - A) * (1 - B) * C * D * (1 - E) * F,
        (1 - A) * (1 - B) * C * D * E * (1 - F),
        (1 - A) * (1 - B) * C * D * E * F,
        (1 - A) * B* (1 - C) * (1 - D) * (1 - E)  *(1 - F),
        (1 - A) * B* (1 - C) * (1 - D) * (1 - E)  * F,
        (1 - A) * B* (1 - C) * (1 - D) * E * (1 - F),
        (1 - A) * B* (1 - C) * (1 - D) * E * F,
        (1 - A) * B* (1 - C) * D * (1 - E) * (1 - F),
        (1 - A) * B* (1 - C) * D * (1 - E) * F,
        (1 - A) * B* (1 - C) * D * E * (1 - F),
        (1 - A) * B* (1 - C) * D * E * F,
        (1 - A) * B* C * (1 - D) * (1 - E) * (1 - F),
        (1 - A) * B* C * (1 - D) * (1 - E) * F,
        (1 - A) * B* C * (1 - D) * E * (1 - F),
        (1 - A) * B* C * (1 - D) * E * F,
        (1 - A) * B* C * D * (1 - E) * (1 - F),
        (1 - A) * B* C * D * (1 - E) * F,
        (1 - A) * B* C * D * E * (1 - F),
        (1 - A) * B* C * D * E * F,
        A * (1 - B) * (1 - C) * (1 - D) * (1 - E) * (1 - F),
        A * (1 - B) * (1 - C) * (1 - D) * (1 - E) * F,
        A * (1 - B) * (1 - C) * (1 - D) * E * (1 - F),
        A * (1 - B) * (1 - C) * (1 - D) * E * F,
        A * (1 - B) * (1 - C) * D * (1 - E) * (1 - F),
        A * (1 - B) * (1 - C) * D * (1 - E) * F,
        A * (1 - B) * (1 - C) * D * E * (1 - F),
        A * (1 - B) * (1 - C) * D * E * F,
        A * (1 - B) * C * (1 - D) * (1 - E) * (1 - F),
        A * (1 - B) * C * (1 - D) * (1 - E) * F,
        A * (1 - B) * C * (1 - D) * E * (1 - F),
        A * (1 - B) * C * (1 - D) * E * F,
        A * (1 - B) * C * D * (1 - E) * (1 - F),
        A * (1 - B) * C * D * (1 - E) * F,
        A * (1 - B) * C * D * E * (1 - F),
        A * (1 - B) * C * D * E * F,
        A * B * (1 - C) * (1 - D) * (1 - E) * (1 - F),
        A * B * (1 - C) * (1 - D) * (1 - E) * F,
        A * B * (1 - C) * (1 - D) * E * (1 - F),
        A * B * (1 - C) * (1 - D) * E * F,
        A * B * (1 - C) * D * (1 - E) * (1 - F),
        A * B * (1 - C) * D * (1 - E) * F,
        A * B * (1 - C) * D * E * (1 - F),
        A * B * (1 - C) * D * E * F,
        A * B * C * (1 - D) * (1 - E) * (1 - F),
        A * B * C * (1 - D) * (1 - E) * F,
        A * B * C * (1 - D) * E * (1 - F),
        A * B * C * (1 - D) * E * F,
        A * B * C * D * (1 - E) * (1 - F),
        A * B * C * D * (1 - E) * F,
        A * B * C * D * E * (1 - F),
        A * B * C * D * E * F
    ], dim=-1)
    return basis


def weighted_light_basis_sum_6(A, B, C, D, E, F, weights, einsum_pattern) -> torch.Tensor:
    return (
        torch.einsum(einsum_pattern, weights[...,0], (1 - A) * (1 - B) * (1 - C) * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,1], (1 - A) * (1 - B) * (1 - C) * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,2], (1 - A) * (1 - B) * (1 - C) * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,3], (1 - A) * (1 - B) * (1 - C) * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,4], (1 - A) * (1 - B) * (1 - C) * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,5], (1 - A) * (1 - B) * (1 - C) * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,6], (1 - A) * (1 - B) * (1 - C) * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,7], (1 - A) * (1 - B) * (1 - C) * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,8], (1 - A) * (1 - B) * C * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,9], (1 - A) * (1 - B) * C * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,10], (1 - A) * (1 - B) * C * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,11], (1 - A) * (1 - B) * C * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,12], (1 - A) * (1 - B) * C * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,13], (1 - A) * (1 - B) * C * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,14], (1 - A) * (1 - B) * C * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,15], (1 - A) * (1 - B) * C * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,16], (1 - A) * B* (1 - C) * (1 - D) * (1 - E)  *(1 - F)) +
        torch.einsum(einsum_pattern, weights[...,17], (1 - A) * B* (1 - C) * (1 - D) * (1 - E)  * F) +
        torch.einsum(einsum_pattern, weights[...,18], (1 - A) * B* (1 - C) * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,19], (1 - A) * B* (1 - C) * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,20], (1 - A) * B* (1 - C) * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,21], (1 - A) * B* (1 - C) * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,22], (1 - A) * B* (1 - C) * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,23], (1 - A) * B* (1 - C) * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,24], (1 - A) * B* C * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,25], (1 - A) * B* C * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,26], (1 - A) * B* C * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,27], (1 - A) * B* C * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,28], (1 - A) * B* C * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,29], (1 - A) * B* C * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,30], (1 - A) * B* C * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,31], (1 - A) * B* C * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,32], A * (1 - B) * (1 - C) * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,33], A * (1 - B) * (1 - C) * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,34], A * (1 - B) * (1 - C) * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,35], A * (1 - B) * (1 - C) * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,36], A * (1 - B) * (1 - C) * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,37], A * (1 - B) * (1 - C) * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,38], A * (1 - B) * (1 - C) * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,39], A * (1 - B) * (1 - C) * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,40], A * (1 - B) * C * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,41], A * (1 - B) * C * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,42], A * (1 - B) * C * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,43], A * (1 - B) * C * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,44], A * (1 - B) * C * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,45], A * (1 - B) * C * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,46], A * (1 - B) * C * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,47], A * (1 - B) * C * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,48], A * B * (1 - C) * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,49], A * B * (1 - C) * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,50], A * B * (1 - C) * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,51], A * B * (1 - C) * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,52], A * B * (1 - C) * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,53], A * B * (1 - C) * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,54], A * B * (1 - C) * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,55], A * B * (1 - C) * D * E * F) +
        torch.einsum(einsum_pattern, weights[...,56], A * B * C * (1 - D) * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,57], A * B * C * (1 - D) * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,58], A * B * C * (1 - D) * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,59], A * B * C * (1 - D) * E * F) +
        torch.einsum(einsum_pattern, weights[...,60], A * B * C * D * (1 - E) * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,61], A * B * C * D * (1 - E) * F) +
        torch.einsum(einsum_pattern, weights[...,62], A * B * C * D * E * (1 - F)) +
        torch.einsum(einsum_pattern, weights[...,63], A * B * C * D * E * F)
    )

####################################################################

def get_regularization_loss(weights, regularizer=None):
    if regularizer is None:
        return 0.0
    elif regularizer == "L2":
        return (1 - weights.pow(2).sum(-1)).pow(2)
    elif regularizer == "abs_sum":
        return (1 - weights.sum(-1).abs()).pow(2)
    elif regularizer == "warp":
        # to enforce a growing norm factor (towards infinity)
        norm_factor = weights[:, -1]
        loss = torch.exp(-norm_factor).mean()
        # print(f"WARP regularization loss: {loss.item()}")
        return loss
    else:
        raise ValueError(f"Unknown regularizer: {regularizer}")
    

def rescale_weights(weights, method=None):
    with torch.no_grad():
        if method is None:
            pass
        elif method == "clip":
            weights.clamp_(-1, 1)
        elif method == "abs_sum":
            abs_sum = weights.sum(dim=-1, keepdim=True).abs()
            weights.div_(abs_sum)
        elif method == "L2":
            l2_norm = weights.norm(p=2, dim=-1, keepdim=True)
            weights.div_(l2_norm)
        else:
            raise ValueError(f"Unknown rescale method: {method}")
        

##########################################################################

def build_binom_table(n: int, k: int, device=None) -> torch.Tensor:
    """Build table C[x, i] = binom(x, i) for x in [0..n-1], i in [0..k]."""
    C = torch.zeros(n, k + 1, dtype=torch.long, device=device)
    C[:, 0] = 1
    for x in range(n):
        for i in range(1, k + 1):
            if i > x:
                C[x, i] = 0
            elif i == x:
                C[x, i] = 1
            else:
                # Pascal recurrence: C(x, i) = C(x-1, i-1) + C(x-1, i)
                C[x, i] = C[x - 1, i - 1] + C[x - 1, i]
    return C

def unrank_combinations_batched(n: int, k: int, ranks: torch.Tensor) -> torch.Tensor:
    """Map ranks in [0, C(n,k)) to k-combinations of {0,...,n-1}, batched.

    Args:
        n: Total number of items.
        k: Combination size.
        ranks: Tensor of integer ranks, shape (...,).

    Returns:
        Tensor of combinations of shape (..., k), dtype long.
    """
    device = ranks.device
    ranks = ranks.clone().reshape(-1)           # (B,)
    B = ranks.numel()

    # Precompute binomial coefficients C[x, i]
    C = build_binom_table(n, k, device=device)  # (n, k+1)

    combos = torch.empty(B, k, dtype=torch.long, device=device)

    # Combinadic unranking: for i = k .. 1, find largest x with C(x, i) <= r
    for i in range(k, 0, -1):
        col = C[:, i]                       # (n,)
        # Broadcast compare: (n,B)
        mask = col.unsqueeze(1) <= ranks.unsqueeze(0)
        # monotone in x ⇒ last True index is desired x
        x = mask.long().sum(dim=0) - 1      # (B,)
        combos[:, i - 1] = x
        ranks = ranks - C[x, i]             # C[x, i] gathered by advanced indexing

    return combos.reshape(*ranks.shape, k)
    
def sample_unique_ranks(total_tuples, sample_size, K, device):
    all_ranks = []
    for _ in range(K):
        r = random.sample(range(total_tuples), sample_size)  # pure Python, no big tensor
        all_ranks.append(r)
    return torch.tensor(all_ranks, dtype=torch.long, device=device)  # (K, sample_size)


def get_combination_indices(n, k, sample_size, num_sets, device):
    """Get unique combination indices for K samples of size sample_size from n items.

    Args:
        n: Total number of items.
        k: Combination size.
        sample_size: Number of unique combinations to sample.
        num_sets: Number of sets of combinations to generate.
        device: Target device for returned tensor.
    """
    total_tuples = math.comb(int(n), k)
    assert sample_size <= total_tuples, (
        f"Not enough unique {k}-tuples: need {sample_size}, have {total_tuples}"
        )
    ranks = sample_unique_ranks(total_tuples, sample_size, num_sets, device)
    comb_indices = unrank_combinations_batched(n, k, 
                                                ranks.view(-1)).view(num_sets, 
                                                                    sample_size, k)
    return comb_indices


def take_tuples(
    x: torch.Tensor,
    tuple_size: int,
    start: int = 0,
    stride_within: int = 1,
    step_between: int = 1,
):
    """
    Take k-tuples from the last dimension of x with full control over:
      - start index
      - stride within each tuple
      - step between tuple starts

    x:            (..., N)
    tuple_size:   k (number of elements per tuple)
    start:        first index used for the first tuple
    stride_within:distance between elements inside a tuple
    step_between: distance between starts of consecutive tuples
                  (if None, defaults to tuple_size * stride_within)

    Returns:
        y with shape (..., tuple_size, num_groups)
        where last two dims are (k, num_groups).
    """
    N = x.size(-1)

    if step_between is None:
        step_between = tuple_size * stride_within

    # Maximum starting index that still fits a full tuple
    max_start = N - (tuple_size - 1) * stride_within
    if max_start <= start:
        raise ValueError("Not enough elements to take a single tuple "
                         f"with start={start}, tuple_size={tuple_size}, stride_within={stride_within}.")

    # All possible starting positions
    starts = torch.arange(start, max_start, step_between, device=x.device)  # (num_groups,)
    num_groups = starts.numel()

    # Offsets inside each tuple
    offsets = torch.arange(tuple_size, device=x.device) * stride_within     # (tuple_size,)

    # Build index matrix: shape (tuple_size, num_groups)
    idx = starts.unsqueeze(0) + offsets.unsqueeze(1)

    # Gather: result (..., tuple_size, num_groups)
    y = x[..., idx]   # advanced indexing broadcasts over leading dims

    return y


def id_to_truth_table(id: torch.Tensor, rank: int, device):
    lut_entries = 1 << rank
    bits = torch.arange(lut_entries, device=device)
    # Reshape for broadcasting: id becomes [..., 1], bits becomes [lut_entries]
    truth01 = (id.unsqueeze(-1) >> (lut_entries - 1 - bits)) & 1
    return 1 - 2 * truth01   # {0,1} → {+1,-1}


def id_to_walsh_coefficients(id: torch.Tensor, rank: int, device):
    truth = id_to_truth_table(id, rank, device=device)
    coeffs = walsh_hadamard_transform(truth, n=rank)
    coeffs = coeffs / (1 << rank)
    return coeffs
