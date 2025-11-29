import torch

from ..functional import (
    WALSH_COEFFICIENTS,
    walsh_hadamard_transform,
)


def initialize_weights_raw(weight_init, out_dim, lut_rank, weight_init_param, device):
    lut_entries = 1 << lut_rank
    if weight_init == "residual":
        # all weights to 0 except for weight number lut_entries - 1, which is set to 5 * weight_init_param
        weights = torch.zeros((out_dim, 1 << lut_entries), device=device)
        weights[:, lut_entries - 1] = 5.0 * weight_init_param
        return weights
    elif weight_init == "random":
        return torch.randn(out_dim, 1 << lut_entries, device=device)
    raise ValueError(f"Unknown weight_init: {weight_init}")


def initialize_weights_walsh(weight_init, out_dim, lut_rank, weight_init_param, device):
    lut_entries = 1 << lut_rank
    if weight_init == "residual":
        n = int((out_dim * weight_init_param) // 2) % (out_dim + 1)
        weights = torch.empty((out_dim, lut_entries), device=device)
        # identity representation, corresponds to Boolean function, which maps MSB (last single variable) to itself
        identity = torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
        transformed_identity = walsh_hadamard_transform(identity, lut_rank, dtype=torch.int32, device=device)
        # randomly sample indices
        indices = torch.randperm(out_dim, device=device)
        weights[indices[:n]] = (1.0 / lut_rank) * transformed_identity.to(torch.float)
        # sample random binary representations
        samples = 1 - 2 * torch.randint(0, 2, (out_dim - n, lut_entries), device=device).to(torch.int32)
        # convert with wh transform
        transformed_samples = walsh_hadamard_transform(samples, lut_rank, device=device, dtype=torch.int32)
        if n < out_dim:
            weights[indices[n:]] = (1.0 / lut_rank) * transformed_samples.to(torch.float)
        return weights
    elif weight_init == "random":
        return torch.randn(out_dim, lut_entries, device=device) * 0.1
    else:
        raise ValueError(weight_init)