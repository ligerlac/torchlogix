"""Functional operations for differentiable logic gate neural networks.

This module provides the core mathematical operations for computing logic gate
operations in a differentiable manner. It includes implementations for binary
operations, vectorized operations, and utility functions for building logic
gate networks.
"""

import numpy as np
import torch
from torch.distributions.gumbel import Gumbel

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |


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

# Attention: these use a different ordering of the operations!
WALSH_COEFFICIENTS = {
    0: (-1, 0, 0, 0),
    1: (+1, 0, 0, 0),
    2: (-0.5, 0.5, 0.5, 0.5),
    3: (0.5, 0.5, 0.5, -0.5),
    4: (0, 0, 0, -1),
    5: (0, 0, 0, +1),
    6: (0.5, -0.5, -0.5, -0.5),
    7: (-0.5, -0.5, -0.5, 0.5),
    8: (-0.5, 0.5, -0.5, -0.5),
    9: (-0.5, -0.5, 0.5, -0.5),
    10: (0, 1, 0, 0),
    11: (0, -1, 0, 0),
    12: (0, 0, 1, 0),
    13: (0, 0, -1, 0),
    14: (0.5, -0.5, 0.5, 0.5),
    15: (0.5, 0.5, -0.5, 0.5),
}


def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)
    return ID_TO_OP[i](a, b)


def bin_op_s(a, b, i_s):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)
    r = torch.zeros_like(a)
    for i in range(16):
        u = ID_TO_OP[i](a, b)
        r = r + i_s[..., i] * u
    return r


# def bin_op_s(a, b, i_s):
#     assert a.shape == b.shape, f"Mismatched shapes: {a.shape}, {b.shape}"
#     r = torch.stack([ID_TO_OP[i](a, b) for i in range(16)], dim=-1)
#     return torch.einsum('...i,...i->...', r, i_s)  # Vectorized multiplication


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


def bin_op_cnn_slow(a, b, i_s):
    """A slower, non-optimized version of bin_op_cnn for clarity."""
    assert a.shape == b.shape, f"Mismatched shapes: {a.shape}, {b.shape}"

    # Compute all 16 logic operations (final dimension = 16)
    r = torch.stack(
        [ID_TO_OP[i](a, b) for i in range(16)], dim=-1
    )  # Shape: [100, 16, 576, 8, 16]

    # Reshape `i_s` to match the required shape for broadcasting
    i_s = i_s.unsqueeze(0).unsqueeze(2)  # Shape: [1, 8, 1, 16, 16]
    # Broadcast to [100, 8, 576, 16, 16]
    i_s = i_s.expand(r.shape[0], -1, r.shape[2], -1, -1)
    i_s = i_s.permute(0, 3, 2, 1, 4)  # Now i_s.shape = [100, 16, 576, 8, 16]
    # Multiply & sum over the logic gates (dimension -1)
    return (r * i_s).sum(dim=-1)  # Shape: [100, 16, 576, 8]


def bin_op_cnn(a, b, i_s):
    assert a.shape == b.shape, f"Mismatched shapes: {a.shape}, {b.shape}"

    # Compute all 16 logic operations vectorized (final dimension = 16)
    r = compute_all_logic_ops_vectorized(a, b)  # Shape: [100, 16, 576, 8, 16]
    
    # Optimized einsum: contract over channel and logic operation dimensions
    # r: [batch, channel, spatial, feature, logic_ops] 
    # i_s: [feature, channel, logic_ops]
    # result: [batch, channel, spatial, feature]
    return torch.einsum('bchdn,dcn->bchd', r, i_s)


def bin_op_cnn_walsh(a, b, i_s):
    assert a.shape == b.shape, f"Mismatched shapes: {a.shape}, {b.shape}"

    A = 2 * a - 1  # Convert to {-1, 1}
    B = 2 * b - 1  # Convert to {-1, 1}

    r = torch.stack([
        torch.ones_like(A),  # 0: 1
        A,                   # 1: A
        B,                   # 2: B
        A * B                # 3: A and B in Walsh basis
    ], dim=-1)

    i_s = i_s.unsqueeze(0).unsqueeze(2)
    i_s = i_s.expand(r.shape[0], -1, r.shape[2], -1, -1)
    i_s = i_s.permute(0, 3, 2, 1, 4)
    return (r * i_s).sum(dim=-1)


##########################################################################


def get_unique_connections(in_dim, out_dim, device="cuda"):
    assert out_dim * 2 >= in_dim, (
        "The number of neurons ({}) must not be smaller than half of the number of inputs "
        "({}) because otherwise not all inputs could be used or considered.".format(
            out_dim, in_dim
        )
    )
    n_max = int(in_dim * (in_dim - 1) / 2)
    assert out_dim <= n_max, (
        "The number of neurons ({}) must not be greater than the number of pair-wise combinations "
        "of the inputs ({})".format(out_dim, n_max)
    )

    x = torch.arange(in_dim).long().unsqueeze(0)

    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # If this was not enough, take pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # If this was not enough, take pairs with offsets >= 2:
    offset = 2
    while out_dim > a.shape[-1]:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)

    perm = torch.randperm(out_dim)

    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b


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
    n_exp = x.size(-1)
    y = x.reshape(-1, n_exp)  # collapse batch dims
    for s in range(n):
        step = 1 << s       # 2^s
        block = step << 1   # 2^(s+1)

        # shape: (batch, n / block, block)
        y = y.view(-1, n_exp // block, block)

        a = y[..., :step]
        b = y[..., step:block]

        y = torch.cat((a + b, a - b), dim=-1)  # still (batch, n/block, block)
        y = y.reshape(-1, n_exp)

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
    x: (..., n) where x[...,k] is the kth scalar input
    returns: (..., 2**n) basis vector 
             matching EXACTLY the example:
             [1, A] \odot [1, B] = [1, B, A, A*B]
             etc.
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


def walsh_basis_2(x: torch.Tensor, indices) -> torch.Tensor:
    A, B = x[..., indices[0]], x[..., indices[1]]
    #A = 1- 2 * a
    #B = 1 - 2 * b
    basis = torch.stack([
        torch.ones_like(A),
        B,
        A,
        A*B
    ], dim=-1)
    return basis


def walsh_basis_3(x: torch.Tensor, indices) -> torch.Tensor:
    A, B, C = x[..., indices[0]], x[..., indices[1]], x[..., indices[2]]
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

def walsh_basis_4(x: torch.Tensor, indices) -> torch.Tensor:
    A, B, C, D = x[..., indices[0]], x[..., indices[1]], x[..., indices[2]], x[..., indices[3]]
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


def walsh_basis_6(x: torch.Tensor, indices) -> torch.Tensor:
    A, B, C, D, E, F = (
        x[..., indices[0]],
        x[..., indices[1]],
        x[..., indices[2]],
        x[..., indices[3]],
        x[..., indices[4]],
        x[..., indices[5]],
    )
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

