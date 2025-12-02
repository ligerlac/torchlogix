"""Functional operations for differentiable logic gate neural networks.

This module provides the core mathematical operations for computing logic gate
operations in a differentiable manner. It includes implementations for binary
operations, vectorized operations, and utility functions for building logic
gate networks.
"""
import math
import numpy as np
import torch

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


##########################################################################


def get_random_unique_connections(in_dim, out_dim, lut_rank=2, device="cuda"):
    """Return unique input index tuples for each output neuron, fully in torch.

    Each output neuron gets lut_rank distinct input indices.
    No two neurons share the same tuple (unordered).

    Args:
        in_dim: Number of input features.
        out_dim: Number of output neurons.
        lut_rank: Number of inputs per neuron.
        device: Target device for returned tensor.

    Returns:
        Tensor of shape (lut_rank, out_dim), dtype int64.
    """
    # Feasibility checks
    assert out_dim * lut_rank >= in_dim, (
        f"Need out_dim * lut_rank >= in_dim to cover all inputs "
        f"({out_dim} * {lut_rank} < {in_dim})."
    )
    n_max = math.comb(in_dim, lut_rank)
    assert out_dim <= n_max, (
        f"Requested {out_dim} unique tuples, but only {n_max} combinations exist."
    )

    # Create input range
    x = torch.arange(in_dim, device=device)

    # Create all lut_rank-combinations in lexicographic order:
    # shape = (n_max, lut_rank)
    combos = torch.combinations(x, r=lut_rank, with_replacement=False)

    # Randomly select out_dim unique tuples
    perm = torch.randperm(combos.size(0), device=device)
    selected = combos[perm[:out_dim]]  # (out_dim, lut_rank)

    # Return shape (lut_rank, out_dim)
    return selected.t().contiguous()


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

def softmax(logits, hard=False, tau=1.0):
    y_soft = torch.nn.functional.softmax(logits / tau, dim=-1)
    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft

def sigmoid(logits, hard=False, tau=1.0):
    y_soft = torch.sigmoid(logits / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


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


def walsh_hadamard_transform(x: torch.Tensor, n: int, dtype=torch.float32, device=None, fast=False) -> torch.Tensor:
    """
    Walsh-Hadamard transform on the last dimension.
    x: (..., 2^n)
    """
    if fast:
        return fwht(x, n)
    else:
        H = hadamard_matrix(1 << n, dtype=dtype, device=device)
        return x @ H
    
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
    if lut_rank == 2:
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


def light_basis_2(A, B) -> torch.Tensor:
    basis = torch.stack([
        (1 - A) * (1 - B),
        (1 - A) * B,
        A * (1 - B),
        A*B
    ], dim=-1)
    return basis

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