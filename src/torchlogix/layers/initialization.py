import torch

from ..functional import (
    WALSH_COEFFICIENTS,
    walsh_hadamard_transform,
)


def initialize_weights_raw(weight_init, out_dim, n_inputs, weight_init_param, device):
    n_exp = 1 << n_inputs
    if weight_init == "residual":
        # all weights to 0 except for weight number n_exp - 1, which is set to 5 * weight_init_param
        weights = torch.zeros((out_dim, 1 << n_exp), device=device)
        weights[:, n_exp - 1] = 5.0 * weight_init_param
        return weights
    elif weight_init == "random":
        return torch.randn(out_dim, 1 << n_exp, device=device)
    raise ValueError(f"Unknown weight_init: {weight_init}")


def initialize_weights_walsh(weight_init, out_dim, n_inputs, weight_init_param, device):
    n_exp = 1 << n_inputs
    if weight_init == "residual":
        n = int((out_dim * weight_init_param) // 2) % (out_dim + 1)
        weights = torch.empty((out_dim, n_exp), device=device)
        # identity representation, corresponds to Boolean function, which maps MSB (last single variable) to itself
        identity = torch.cat([torch.zeros(n_exp // 2), torch.ones(n_exp - n_exp // 2)]).to(dtype=torch.int32, device=device)
        transformed_identity = walsh_hadamard_transform(identity, n_inputs, dtype=torch.int32, device=device)
        # randomly sample indices
        indices = torch.randperm(out_dim, device=device)
        weights[indices[:n]] = (1.0 / n_inputs) * transformed_identity.to(torch.float)
        # sample random binary representations
        samples = 1 - 2 * torch.randint(0, 2, (out_dim - n, n_exp), device=device).to(torch.int32)
        # convert with wh transform
        transformed_samples = walsh_hadamard_transform(samples, n_inputs, device=device, dtype=torch.int32)
        if n < out_dim:
            weights[indices[n:]] = (1.0 / n_inputs) * transformed_samples.to(torch.float)
        return weights
    elif weight_init == "random":
        return torch.randn(out_dim, n_exp, device=device) * 0.1
    else:
        raise ValueError(weight_init)