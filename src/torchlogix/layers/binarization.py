from typing import Union
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F

from ..functional import sigmoid, gumbel_sigmoid


def setup_binarization(thresholds, binarization: str, **binarization_kwargs):
    bin_dict = {
        "dummy": DummyBinarization,
        "uniform": FixedBinarization,
        "distributive": FixedBinarization,
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
    def __init__(self, thresholds: Tensor, feature_dim=-2, **kwargs):
        super().__init__()
        self.register_buffer('thresholds', thresholds)
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Subclasses must implement forward."""
        pass
    
    @staticmethod
    def get_uniform_thresholds(data_set: Tensor, num_bits: int, feature_wise: bool) -> Tensor:
        min_value = data_set.min(dim=0)[0] if feature_wise else data_set.min()
        max_value = data_set.max(dim=0)[0] if feature_wise else data_set.max()
        return min_value.unsqueeze(-1) + torch.arange(1, num_bits+1).unsqueeze(0) * (
            (max_value - min_value) / (num_bits + 1)).unsqueeze(-1)
    
    @staticmethod
    def get_distributive_thresholds(data_set: Tensor, num_bits: int, feature_wise: bool) -> Tensor:
        data = torch.sort(data_set.flatten())[0] if not feature_wise else torch.sort(data_set, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(num_bits+1)) for i in range(1, num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))
    
    @staticmethod
    def get_initial_thresholds(data_set: Tensor, num_bits: int, feature_wise: bool, method: str = "uniform") -> Tensor:
        if method == "dummy":
            return None
        elif method in ["uniform", "learnable"]:
            return Binarization.get_uniform_thresholds(data_set, num_bits, feature_wise)
        elif method == "distributive":
            return Binarization.get_distributive_thresholds(data_set, num_bits, feature_wise)
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
        x = (x > self.thresholds).float()
        return merge_dim_with_last(x, self.feature_dim)
        

class DummyBinarization(Binarization):
    """ Dummy binarization module that does nothing."""
    def __init__(self, **kwargs):
        super().__init__(None)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.float()
    

class LearnableBinarization(Binarization):
    def __init__(self, 
                 thresholds: Tensor = None, 
                 feature_wise=True, 
                 feature_dim=-2,
                 temperature_sampling=0.1, 
                 temperature_softplus=1.0,
                 forward_sampling="soft", 
                 **kwargs):
        self.forward_sampling = forward_sampling
        self.feature_wise = feature_wise
        super().__init__(thresholds, feature_dim)
        self.temperature_sampling = temperature_sampling
        self.temperature_softplus = temperature_softplus  
        self._frozen = False  # switch to control hard/soft behavior
        diffs = torch.diff(thresholds, 
                           prepend=thresholds.new_zeros(*thresholds.shape[:-1], 1), dim=-1)
        self.raw_diffs = torch.nn.Parameter(diffs)

    def get_thresholds(self):
        if self._frozen:
            return torch.cumsum(self.raw_diffs, dim=-1)
        else:
            diffs_pos = self.temperature_softplus * F.softplus(self.raw_diffs / self.temperature_softplus)
            return torch.cumsum(diffs_pos, dim=-1)

    def freeze_thresholds(self):
        """I dont get this function"""
        with torch.no_grad():
            thresholds = self.get_thresholds()  #.round()
            diffs = torch.diff(thresholds, 
                               prepend=thresholds.new_zeros(*thresholds.shape[:-1], 1))
            self.raw_diffs.data = diffs
            self.raw_diffs.detach_()
        # self.raw_diffs.requires_grad = False
        self._frozen = True

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
        if self._frozen:
            # Hard thermometer encoding
            outputs = self._sample_eval(x, thresholds)
        else:
            outputs = self._sample_train(x, thresholds)
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
