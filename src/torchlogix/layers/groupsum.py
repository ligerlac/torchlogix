import torch
import torch.library as _tlib

from torchlogix.packbitstensor import PackBitsTensor


@_tlib.custom_op("torchlogix::group_sum", mutates_args=())
def _group_sum_op(x: torch.Tensor, k: int, tau: float, beta: float) -> torch.Tensor:
    result = x.reshape(*x.shape[:-1], k, x.shape[-1] // k).sum(-1).float()
    if beta != 0.0:
        result = result + beta
    if tau != 1.0:
        result = result / tau
    return result


@_group_sum_op.register_fake
def _group_sum_fake(x, k, tau, beta):
    return x.new_empty(list(x.shape[:-1]) + [k], dtype=torch.float32)


from torch._decomp import register_decomposition as _reg_decomp


@_reg_decomp(torch.ops.torchlogix.group_sum.default)
def _group_sum_decomp(x: torch.Tensor, k: int, tau: float, beta: float) -> torch.Tensor:
    result = x.reshape(*x.shape[:-1], k, x.shape[-1] // k).sum(-1).float()
    if beta != 0.0:
        result = result + beta
    if tau != 1.0:
        result = result / tau
    return result


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

        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                result = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1).astype(np.float32)
                if self.beta != 0.0:
                    result = result + self.beta
                if self.tau != 1.0:
                    result = result / self.tau
                return result
        except ImportError:
            pass

        return torch.ops.torchlogix.group_sum(x, self.k, float(self.tau), float(self.beta))

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)

    def set_export_mode(self, export_mode: bool):
        self.eval()
        self.export_mode = export_mode
