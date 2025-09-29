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


def gumbel_sigmoid_old(logits, tau=1.0, hard=False, threshold=0.5):

    """
    Samples from the Gumbel-Sigmoid distribution with an optional hard gate at the selected threshold.
    
    Gumbel Sigmoid is equivalent to gumbel softmax for two classes with one class being 0
    i.e. gumbel_sigmoid = e^([a+gumbel1]/t) / [e^([a+gumbel1]/t) + e^(gumbel2/t)] = sigm([a+gumbel1-gumbel2]/t)

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      temp: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    # Temperature must be positive.
    if tau <= 0:
        raise ValueError("Temperature must be positive")

    # Sample Gumbel noise. The difference of two Gumbels is equivalent to a Logistic distribution.
    gumbel_noise = Gumbel(0, 1).sample(logits.shape).to(logits.device) - \
                   Gumbel(0, 1).sample(logits.shape).to(logits.device)
    
    # Apply the reparameterization trick
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        # Straight-Through Estimator
        y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft
    
    return y_soft



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
