import torch

from torchlogix.packbitstensor import PackBitsTensor


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0, beta=0.0, device="cpu", export_mode=False):
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
        self.export_mode = export_mode

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."

        return (
            (x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) + self.beta) / self.tau
        )

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)
    
    def set_export_mode(self, export_mode: bool):
        self.eval()
        self.export_mode = export_mode