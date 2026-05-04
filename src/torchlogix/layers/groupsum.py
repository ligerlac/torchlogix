import torch

from torchlogix.packbitstensor import PackBitsTensor


def setup_group_sum(group_sum_method, **group_sum_kwargs):
    if group_sum_method == "groupsum":
        return GroupSum(**group_sum_kwargs)
    elif group_sum_method == "learnable_affine":
        return LearnableGroupAffine(**group_sum_kwargs)
    elif group_sum_method == "learnable_linear":
        return LearnableGroupLinear(**group_sum_kwargs)
    else:
        raise ValueError(f"Unknown groupsum method: {group_sum_method}")
    

class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0, beta=0.0):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."

        return (
            (x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) + self.beta) / self.tau
        )

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)


class LearnableGroupAffine(torch.nn.Module):
    """
    The continuous affine GroupSum module, as in "Differentiable Weightless Controllers", Kresse et. al.
    """

    def __init__(self, k: int, init_a=-0.6931, init_b=0.0):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param init_a: initial value for parameter a
        :param init_b: initial value for parameter b
        :param device:
        """
        super().__init__()
        self.k = k
        self.a = torch.nn.Parameter(torch.full((k,), init_a))
        self.b = torch.nn.Parameter(torch.full((k,), init_b))

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return NotImplementedError("ContinuousGroupAffine does not support PackBitsTensor inputs.")

        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."
        n_k = x.shape[-1] // self.k
        x = x.reshape(*x.shape[:-1], self.k, n_k).sum(-1)  # popcount
        x = 2 * x / n_k - 1.0  # normalize to interval [-1, 1]
        x = x * torch.exp(self.a) + self.b  # scale and shift
        return x

    def extra_repr(self):
        return "k={}".format(self.k)
    

class LearnableGroupLinear(torch.nn.Module):
    """
    The continuous GroupSum module with a linear transformation.
    """

    def __init__(self, k: int):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param device:
        """
        super().__init__()
        self.k = k
        self.linear = torch.nn.Linear(k, k)

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return NotImplementedError("ContinuousGroupSumLinear does not support PackBitsTensor inputs.")

        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."
        n_k = x.shape[-1] // self.k
        x = x.reshape(*x.shape[:-1], self.k, n_k).sum(-1)  # popcount
        x = 2 * x / n_k - 1.0  # normalize to interval [-1, 1]
        x = self.linear(x)  # apply linear transformation
        return x

    def extra_repr(self):
        return "k={}".format(self.k)
    