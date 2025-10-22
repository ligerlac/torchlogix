import torch

from torchlogix.packbitstensor import PackBitsTensor


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0, beta=0.0, device="cuda"):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."

        # # Handle both batched and non-batched inputs
        # if len(x.shape) == 2:  # Shape is [N, features]
        #     # Original non-batched case
        #     return (
        #         x.reshape(x.shape[0], self.k, x.shape[-1] // self.k).sum(-1) / self.tau
        #         + self.beta
        #     )
        # elif len(x.shape) == 3:  # Shape is [batch, N, features]
        #     # Batched case
        #     return (
        #         x.reshape(x.shape[0], x.shape[1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau
        #         + self.beta
        #     )
        # else:
        #     raise ValueError(f"Unsupported input shape: {x.shape}")
        return (
            (x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) + self.beta) / self.tau
        )

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)