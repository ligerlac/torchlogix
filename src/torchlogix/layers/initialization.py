import torch

from ..functional import (
    WALSH_COEFFICIENTS,
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
        # chose randomly from walsh_coefficients, but prefer id=n_exp - 1
        walsh_coefficients_tensor = torch.tensor(list(WALSH_COEFFICIENTS.values()), device=device)
        weights = walsh_coefficients_tensor[
            torch.randint(0, n_exp, (out_dim,), device=device)
        ]
        n = int((out_dim * weight_init_param) // 2)
        # set percentage of weights to id=n_exp - 1 (pick index randomly)
        indices = torch.randperm(out_dim, device=device)
        weights[indices[:n]] = walsh_coefficients_tensor[n_exp - 1]
        return weights
    elif weight_init == "random":
        return torch.randn(out_dim, n_exp, device=device) * 0.1
    else:
        raise ValueError(weight_init)