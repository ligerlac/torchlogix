"""Utility functions for torchlogix layers."""

import numpy as np


def apply_luts_vectorized(a, b, lut_ids):
    """
    Apply 2-input LUTs in parallel using vectorized boolean operations.

    This function works with:
    - torch.Tensor (converted to numpy)
    - numpy.ndarray
    - da4ml's FixedVariableArray

    For FixedVariableArray, this converts to numpy, applies operations,
    then converts back to preserve the computation graph building.

    Args:
        a: First inputs, shape (..., N)
        b: Second inputs, shape (..., N)
        lut_ids: LUT IDs for each neuron, numpy array of shape (N,)
                 Values in range [0, 15] representing 2-input boolean functions

    Returns:
        Output in same array type as inputs, shape (..., N)
    """
    # Determine array type and convert if necessary using duck typing
    # Check for solver_options attribute (da4ml's FixedVariableArray)
    if hasattr(a, 'solver_options'):
        # Array-like with solver_options (e.g., FixedVariableArray)
        # Save solver_options and type for wrapping result back
        solver_options = a.solver_options
        array_type = type(a)
        is_torch = False
        # Convert to numpy for operations (preserves special objects like FixedVariable)
        a, b = np.array(a), np.array(b)
        # Don't convert to bool - special objects may support boolean operations
        result = np.empty_like(a)
    elif hasattr(a, 'cpu'):
        # torch.Tensor - convert to numpy then bool
        solver_options = None
        is_torch = True
        a_np = a.cpu().detach().numpy() if a.requires_grad else a.cpu().numpy()
        b_np = b.cpu().detach().numpy() if b.requires_grad else b.cpu().numpy()
        original_device = a.device
        a, b = a_np.astype(bool), b_np.astype(bool)
        result = np.empty_like(a, dtype=bool)
    else:
        # numpy.ndarray - convert to bool
        solver_options = None
        is_torch = False
        a = a.astype(bool)
        b = b.astype(bool)
        result = np.empty_like(a, dtype=bool)

    # LUT truth table map
    _map = [
        lambda a, b: 0,              # FALSE
        lambda a, b: a & b,          # AND
        lambda a, b: a & ~b,         # A AND NOT B
        lambda a, b: a,              # A
        lambda a, b: b & ~a,         # B AND NOT A
        lambda a, b: b,              # B
        lambda a, b: a ^ b,          # XOR
        lambda a, b: a | b,          # OR
        lambda a, b: ~(a | b),       # NOR
        lambda a, b: ~(a ^ b),       # XNOR
        lambda a, b: ~b,             # NOT B
        lambda a, b: ~(b & ~a),      # A OR NOT B
        lambda a, b: ~a,             # NOT A
        lambda a, b: ~(a & ~b),      # B OR NOT A
        lambda a, b: ~(a & b),       # NAND
        lambda a, b: 1,              # TRUE
    ]

    # For each LUT type, apply it where needed
    for lut_id in range(16):
        mask = lut_ids == lut_id
        result[..., mask] = _map[lut_id](a[..., mask], b[..., mask])

    # Convert back to original type if needed
    if solver_options is not None:
        # Result already contains special objects, wrap using original type
        result = array_type(result, solver_options=solver_options)
    elif is_torch:
        # Convert bool to float for torch
        import torch
        result = torch.from_numpy(result.astype(np.float32)).to(original_device)
    # For plain numpy, result is already correct type (bool)

    return result


def pad_generic(x, padding):
    """
    Type-generic 2D padding for arrays.

    Works with torch.Tensor, numpy.ndarray, and FixedVariableArray.

    Args:
        x: Input array of shape (batch, channels, H, W)
        padding: Padding amount for all sides

    Returns:
        Padded array in same type as input
    """
    if padding == 0:
        return x

    # Dispatch based on type
    if hasattr(x, 'pad'):  # torch.Tensor
        import torch.nn.functional as F
        return F.pad(x, (padding, padding, padding, padding),
                    mode='constant', value=0)
    else:  # numpy or FixedVariableArray
        return np.pad(
            x,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )


def stack_luts(lut_list):
    """
    Stack LUT IDs from list of tensors into numpy array.

    This is used for configuration data (not traced computation).

    Args:
        lut_list: List of tensors with LUT IDs

    Returns:
        Stacked numpy array
    """
    numpy_luts = []
    for lut in lut_list:
        if hasattr(lut, 'cpu'):
            numpy_luts.append(lut.cpu().detach().numpy())
        else:
            numpy_luts.append(lut)
    return np.stack(numpy_luts, axis=0)


def apply_luts_conv_level(a, b, lut_ids, batch, K, P, N):
    """
    Apply LUTs for one convolutional tree level with proper reshaping.

    Args:
        a: First inputs, shape (batch, K, P, N)
        b: Second inputs, shape (batch, K, P, N)
        lut_ids: LUT IDs, numpy array of shape (N, K)
        batch: Batch size
        K: Number of kernels
        P: Number of spatial positions
        N: Number of nodes at this level

    Returns:
        Result in same type as inputs, shape (batch, K, P, N)
    """
    # Reshape for vectorized application
    # Use np.transpose for type-generic transpose (works with FixedVariableArray)
    a_flat = np.transpose(a, (0, 2, 1, 3)).reshape(batch * P, K * N)
    b_flat = np.transpose(b, (0, 2, 1, 3)).reshape(batch * P, K * N)
    lut_flat = lut_ids.T.flatten()

    # Apply LUTs
    result = apply_luts_vectorized(a_flat, b_flat, lut_flat)

    # Reshape back
    result = result.reshape(batch, P, K, N)
    result = np.transpose(result, (0, 2, 1, 3))

    return result
