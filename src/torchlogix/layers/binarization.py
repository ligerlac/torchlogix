from typing import Union, List
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F

from ..functional import sigmoid, gumbel_sigmoid


def setup_binarization(thresholds, binarization: str, **binarization_kwargs):
    bin_dict = {
        "dummy": DummyBinarization,
        "fixed": FixedBinarization,
        "soft": SoftBinarization,
        "learnable": LearnableBinarization
    }
    if binarization not in bin_dict:
        raise ValueError(
            f"Unsupported binarization method: {binarization}. "
            f"Choose from {list(bin_dict.keys())}."
        )
    bin_cls = bin_dict[binarization]
    return bin_cls(thresholds=thresholds, **binarization_kwargs)


class Binarization(torch.nn.Module, ABC):
    """Abstract base class for binarization modules."""
    def __init__(
            self,
            thresholds: Tensor | List,
            feature_dim=-2,
            **kwargs
        ):
        super().__init__()
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.register_buffer('thresholds', thresholds)
        self.feature_dim = feature_dim

    def get_thresholds(self):
        """Return thresholds. Subclasses can override for learnable thresholds."""
        return self.thresholds
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Subclasses must implement forward."""
        pass
    
    @staticmethod
    def get_uniform_thresholds(data_set: Tensor, num_bits: int, one_per: str) -> Tensor:
        print(f"Data set shape for uniform threshold calculation (per={one_per}): {data_set.shape}")
        if one_per == "feature":
            min_value = data_set.min(dim=0)[0]
            max_value = data_set.max(dim=0)[0]
        elif one_per == "channel":
            # For conv: (batch_size, num_channels, h, w)
            # We want min/max per channel across all other dimensions
            batch_size, num_channels = data_set.shape[:2]
            # Reshape to (num_channels, -1) to flatten all non-channel dims
            reshaped = data_set.transpose(0, 1).reshape(num_channels, -1)
            min_value = reshaped.min(dim=1)[0]  # (num_channels,)
            max_value = reshaped.max(dim=1)[0]  # (num_channels,)
        elif one_per == "global":
            min_value = data_set.min()
            max_value = data_set.max()
        else:
            raise ValueError(f"one_per must be 'feature', 'channel', or 'global'. Got {one_per}.")
        threshs = min_value.unsqueeze(-1) + torch.arange(1, num_bits+1).unsqueeze(0) * (
            (max_value - min_value) / (num_bits + 1)).unsqueeze(-1)
        print(f"Uniform thresholds shape: {threshs.shape}")
        print(f"threshs =\n{threshs}")
        return threshs
        
    @staticmethod
    def get_distributive_thresholds(
        data_set: Tensor,
        num_bits: int,
        one_per: str
    ) -> Tensor:
        """
        Compute distributive (quantile-based) thresholds.

        one_per:
            - "global":   one set of thresholds for entire tensor
            - "feature":  per-feature thresholds (last dimension)
            - "channel":  per-channel thresholds (dim=1, conv tensors)
        """
        if one_per == "global":
            # Flatten everything
            data = torch.sort(data_set.flatten())[0]  # (N,)
            indices = torch.tensor(
                [int(data.numel() * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=data.device
            )
            thresholds = data[indices]  # (num_bits,)
            return thresholds

        elif one_per == "feature":
            # Feature = last dimension
            # Shape: (..., F)
            data = torch.sort(data_set, dim=0)[0]  # sort along batch dimension
            n = data.shape[0]

            indices = torch.tensor(
                [int(n * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=data.device
            )

            thresholds = data[indices, ...]  # (num_bits, F)
            return thresholds.permute(1, 0)  # (F, num_bits)

        elif one_per == "channel":
            # Expected shape: (batch, channels, ...)
            batch_size, num_channels = data_set.shape[:2]

            # Move channels first and flatten everything else
            reshaped = data_set.transpose(0, 1).reshape(num_channels, -1)
            sorted_data = torch.sort(reshaped, dim=1)[0]  # (C, N)

            n = sorted_data.shape[1]
            indices = torch.tensor(
                [int(n * i / (num_bits + 1)) for i in range(1, num_bits + 1)],
                device=sorted_data.device
            )

            thresholds = sorted_data[:, indices]  # (C, num_bits)
            return thresholds

        else:
            raise ValueError(
                f"one_per must be 'global', 'feature', or 'channel'. Got {one_per}."
            )

    
    @staticmethod
    def get_initial_thresholds(data_set: Tensor, num_bits: int, one_per: str, method: str = "uniform") -> Tensor:
        assert one_per in ["feature", "channel", "global"], "one_per must be 'feature', 'channel', or 'global'."
        assert method in ["uniform", "distributive"], "method must be 'uniform' or 'distributive'"
        if method == "uniform":
            return Binarization.get_uniform_thresholds(data_set, num_bits, one_per)
        elif method == "distributive":
            return Binarization.get_distributive_thresholds(data_set, num_bits, one_per)
        else:
            raise ValueError(f"Unknown threshold initialization method: {method}.")


class FixedBinarization(Binarization):
    """Binarization with fixed (non-learnable) thresholds."""
    def __init__(self, thresholds: Tensor, feature_dim=-2, **kwargs):
        super().__init__(thresholds, feature_dim, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.thresholds is None:
            raise ValueError('Need to fit before calling apply')
        x = x.unsqueeze(-1)
        if self.thresholds.dim() == 2 and x.dim() == 5:  # Conv with channel-wise thresholds
            # thresholds: (num_channels, num_bits)
            # x: (batch, channels, h, w, 1)
            # Reshape to (1, num_channels, 1, 1, num_bits)
            thresholds = self.thresholds.view(1, -1, 1, 1, self.thresholds.shape[1])
            x = (x > thresholds).float()
        else:
            x = (x > self.thresholds).float()
        return merge_dim_with_last(x, self.feature_dim)
        

class DummyBinarization(Binarization):
    """ Dummy binarization module that does nothing."""
    def __init__(self, **kwargs):
        super().__init__(None)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.float()
    

class SoftBinarization(Binarization):
    """ Soft binarization with fixed thresholds using sigmoid."""
    def __init__(self, thresholds: Tensor, temperature=0.1, feature_dim=-2, **kwargs):
        super().__init__(thresholds, feature_dim, **kwargs)
        self.temperature_sampling = temperature
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        if self.thresholds.dim() == 2 and x.dim() == 5:  # Conv with channel-wise thresholds
            # thresholds: (num_channels, num_bits)
            # x: (batch, channels, h, w, 1)
            # Reshape to (1, num_channels, 1, 1, num_bits)
            thresholds = self.thresholds.view(1, -1, 1, 1, self.thresholds.shape[1])
        else:
            thresholds = self.thresholds
        if self.training:
            x = sigmoid((x - thresholds) / self.temperature_sampling)
        else:
            x = (x > thresholds).to(dtype=torch.float32)
        return merge_dim_with_last(x, self.feature_dim)
    

class LearnableBinarization(Binarization):
    def __init__(
        self, 
        thresholds: Tensor | List,
        feature_dim=-2,
        temperature_sampling=0.1,
        temperature_softplus=0.1,
        forward_sampling="soft", 
        max_grad_norm=0.001,
        **kwargs
    ):
        self.forward_sampling = forward_sampling
        super().__init__(thresholds, feature_dim)
        self.temperature_sampling = temperature_sampling
        self.temperature_softplus = temperature_softplus
        diffs = torch.diff(self.thresholds, 
                           prepend=self.thresholds.new_zeros(*self.thresholds.shape[:-1], 1), dim=-1)
        self.raw_diffs = torch.nn.Parameter(diffs)
        self.raw_diffs.register_hook(self._clip_grad)
        self._max_grad_norm = max_grad_norm

    def _clip_grad(self, grad):
        norm = grad.norm()
        if norm > self._max_grad_norm:
            grad = grad * (self._max_grad_norm / (norm + 1e-6))
        return grad
            
    def get_thresholds(self):
        if self.training:
            # first diff can be negative: global shift
            first_diff = self.raw_diffs[..., :1]  # unconstrained

            # remaining diffs are positive
            if self.raw_diffs.shape[-1] > 1:
                rest_diffs = (self.temperature_softplus + 1e-6) * F.softplus(
                    self.raw_diffs[..., 1:] / (self.temperature_softplus + 1e-6)
                )
                diffs_pos = torch.cat([first_diff, rest_diffs], dim=-1)
            else:
                diffs_pos = first_diff

            thresholds = torch.cumsum(diffs_pos, dim=-1)

        else:
            thresholds = torch.cumsum(self.raw_diffs, dim=-1)

        return thresholds        

    def _sample_train(self, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid/gumbel_sigmoid based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return sigmoid(x - thresholds, tau=self.temperature_sampling, hard=False)
        elif self.forward_sampling == "hard":
            return sigmoid(x - thresholds, tau=self.temperature_sampling, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_sigmoid(x - thresholds, tau=self.temperature_sampling, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_sigmoid(x - thresholds, tau=self.temperature_sampling, hard=True)

    def _sample_eval(self, x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """Threshold for discrete output."""
        return (x > thresholds).to(dtype=torch.float32)

    def forward(self, x):
        x = x.unsqueeze(-1)
        thresholds = self.get_thresholds()
        if thresholds.dim() == 2 and x.dim() == 5:  # Conv with channel-wise thresholds
            # thresholds: (num_channels, num_bits)
            # x: (batch, channels, h, w, 1)
            # Reshape to (1, num_channels, 1, 1, num_bits)
            thresholds = thresholds.view(1, -1, 1, 1, thresholds.shape[1])
        if self.training:
            # Hard thermometer encoding
            outputs = self._sample_train(x, thresholds)
        else:
            outputs = self._sample_eval(x, thresholds)
        return merge_dim_with_last(outputs, self.feature_dim)
    

def merge_dim_with_last(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Merge dimension k with the last dimension of x.

    Input shape:  (d0, d1, ..., d{k}, ..., d{n-2}, d{n-1})
    Output shape: (d0, ..., d{k}*d{n-1}, ..., d{n-2})
    i.e., last dim is folded into dim k, and the last dim disappears.

    The order of the other dimensions is preserved.
    """
    n = x.ndim
    if n < 2:
        raise ValueError("Need at least 2 dimensions to merge with last.")
    if k < 0:
        k += n
    if not (0 <= k < n - 1):
        raise ValueError(f"k must be in [0, {n-2}] (cannot be the last dim). Got {k}.")

    last = n - 1

    # Permute so dims become: [0..k-1, k, last, k+1..last-1]
    perm = list(range(n))
    perm.pop(last)          # remove last
    perm.insert(k + 1, last)  # insert last right after k

    y = x.permute(*perm)

    # Now k and k+1 are adjacent: (.., d_k, d_last, ..)
    shape = list(y.shape)
    shape[k] = shape[k] * shape[k + 1]   # merge
    shape.pop(k + 1)                     # drop the extra dim
    y = y.reshape(*shape)

    return y
