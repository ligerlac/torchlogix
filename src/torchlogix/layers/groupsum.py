import math
from numbers import Real

import torch

from ..modes import ExportableModule


class GroupSum(ExportableModule):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau=1.0, beta=0.0, device="cpu", export_mode=False):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: scalar or length-k sequence of divisors for the group sums.
        :param beta: scalar or length-k sequence of offsets added before division.
        :param device:
        """
        super().__init__()
        self.k = k
        self.register_buffer("_scale_anchor", torch.empty(0), persistent=False)
        self.tau = tau
        self.beta = beta
        self.device = device
        self.export_mode = export_mode

    @staticmethod
    def _normalize_scale(value, k, name):
        # Preserve Python scalars as-is: existing callers can still read and
        # assign .tau/.beta without changing their state_dict or export path.
        if isinstance(value, Real) or (isinstance(value, torch.Tensor) and value.ndim == 0):
            return value
        values = torch.as_tensor(value)
        if values.ndim != 1 or values.numel() != k:
            raise ValueError(f"{name} must be a scalar or a length-{k} vector")
        result = tuple(float(v) for v in values.tolist())
        if any(not math.isfinite(v) for v in result):
            raise ValueError(f"{name} values must be finite")
        if name == "tau" and any(v <= 0 for v in result):
            raise ValueError("tau values must be positive")
        return result

    def _set_scale(self, value, name):
        value = self._normalize_scale(value, self.k, name)
        if isinstance(value, tuple):
            self.register_buffer(
                f"_{name}_vector",
                torch.tensor(
                    value, dtype=self._scale_anchor.dtype, device=self._scale_anchor.device
                ),
            )
            object.__setattr__(self, f"_{name}_scalar", None)
        else:
            self.register_buffer(f"_{name}_vector", None, persistent=False)
            object.__setattr__(self, f"_{name}_scalar", value)

    @property
    def tau(self):
        return self._buffers.get("_tau_vector") if self._tau_scalar is None else self._tau_scalar

    @tau.setter
    def tau(self, value):
        self._set_scale(value, "tau")

    @property
    def beta(self):
        return self._buffers.get("_beta_vector") if self._beta_scalar is None else self._beta_scalar

    @beta.setter
    def beta(self, value):
        self._set_scale(value, "beta")

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."

        result = x.reshape(x.shape[:-1] + (self.k, x.shape[-1] // self.k)).sum(-1)
        beta = self.beta
        tau = self.tau
        if (isinstance(beta, torch.Tensor) and beta.ndim == 1) or beta != 0.0:
            result = result + beta
        if (isinstance(tau, torch.Tensor) and tau.ndim == 1) or tau != 1.0:
            result = result / tau
        return result

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)

